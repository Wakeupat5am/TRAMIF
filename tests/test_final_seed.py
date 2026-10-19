import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from scripts import train_final_seed as t


class TestFinalSeed(unittest.TestCase):
    def test_seed_and_execution_guards_before_io(self):
        for seed in (42,47,True,'43'):
            with self.assertRaises(ValueError):t.validate_seed(seed)
        with patch.object(t,'verify_preflight',side_effect=AssertionError('no reads')):
            with self.assertRaises(PermissionError):t.train(43,'sbsmi','absent','bad')

    def test_fresh_seeded_initialization_and_shapes(self):
        previous=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for view in t.VIEWS:
                a=t.make_model(43,view).eval();b=t.make_model(43,view).eval();c=t.make_model(44,view).eval()
                self.assertTrue(torch.equal(a.classifier.weight,b.classifier.weight))
                self.assertFalse(torch.equal(a.classifier.weight,c.classifier.weight))
                before=copy.deepcopy(a.state_dict())
                with torch.inference_mode():self.assertEqual(a(torch.zeros(2,1,64,64)).shape,(2,51))
                self.assertTrue(all(torch.equal(before[k],v) for k,v in a.state_dict().items()))
                self.assertTrue(all(p.grad is None for p in a.parameters()))
        finally:torch.set_num_threads(previous)

    def test_shuffle_coverage_seed_specific(self):
        data=list(range(18061));a,_=t.make_loader(data,43,False);b,_=t.make_loader(data,43,False);c,_=t.make_loader(data,44,False)
        aa=[i for batch in a for i in batch.tolist()];bb=[i for batch in b for i in batch.tolist()];cc=[i for batch in c for i in batch.tolist()]
        self.assertEqual(aa,bb);self.assertNotEqual(aa,cc);self.assertEqual(sorted(aa),data)
        self.assertEqual(len(a),565)

    def test_chronology_mapping_duplicates_fail_closed(self):
        mapping={f'f{i}':i for i in range(51)}
        row=dict(sha='a',timestamp='2019-08-01 00:00:00+00:00',split='train',family='f0')
        t.validate_train_rows([row],mapping,1)
        for change in (dict(timestamp='2020-02-01 00:00:00+00:00'),dict(timestamp='2019-08-01'),dict(split='validation'),dict(family='bad')):
            with self.assertRaises(ValueError):t.validate_train_rows([dict(row,**change)],mapping,1)
        with self.assertRaises(ValueError):t.validate_train_rows([row,row],mapping,2)

    def test_checkpoint_contract_json_tuple_parity_and_optimizer_no_step(self):
        model=nn.Linear(2,3)
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('no optimization')):
            optimizer=t.make_optimizer(model)
            config=dict(run_spec=dict(t.RUN_SPEC,seed=43),adam_defaults=optimizer.defaults)
            canonical=json.loads(t.json_bytes(config));self.assertEqual(t.json_bytes(config),t.json_bytes(canonical))
            payload=dict(epoch=100,config=config,config_sha256=t.sha256(t.json_bytes(config)),model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict())
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'epoch_100.pt';digest=t.save_checkpoint(path,payload)
                self.assertEqual(digest,t.sha256(path.read_bytes()))
                with self.assertRaises(FileExistsError):t.save_checkpoint(path,payload)
                loaded=torch.load(io.BytesIO(path.read_bytes()),weights_only=True)
                self.assertEqual(t.json_bytes(loaded['config']),t.json_bytes(canonical))

    def test_missing_or_ambiguous_run_no_automatic_latest(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(t,'ROOT',Path(tmp)):
            with self.assertRaises(ValueError):t.discover(43,'sbsmi')

    def test_complete_contract_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(t,'ROOT',Path(tmp)),patch.object(t,'source_hashes',return_value={'synthetic':'pinned'}),patch.object(t,'FinalTrainDataset') as dataset:
            dataset.return_value.binding={'K':51,'cutoff_exclusive':'2020-02'}
            run=Path(tmp)/t.OUTPUT/'seed_43/sbsmi/synthetic';run.mkdir(parents=True)
            model=nn.Linear(2,3);optimizer=t.make_optimizer(model)
            config=dict(seed=43,view='sbsmi',run_spec=dict(t.RUN_SPEC,seed=43),dataset=dataset.return_value.binding,code_sha256={'synthetic':'pinned'},adam_defaults=optimizer.defaults)
            h=t.sha256(t.json_bytes(config));t.write_new_json(run/'config.json',config)
            pt=t.save_checkpoint(run/'epoch_100.pt',dict(epoch=100,config=config,config_sha256=h,model_state_dict=model.state_dict()))
            epochs=[dict(epoch=i,sha256=pt) for i in range(1,101)]
            t.write_new_json(run/'epoch_100.json',epochs[-1]);t.write_new_json(run/'COMPLETE.json',dict(epochs=epochs,selected=epochs[-1],config_sha256=h))
            binding=t.discover(43,'sbsmi')
            with patch.object(t,'make_model',return_value=nn.Linear(2,3)):
                loaded,_,_=t.load_frozen_run(binding['run_dir'],43,'sbsmi',binding['complete_sha256'])
            self.assertFalse(loaded.training);self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            (run/'epoch_100.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):t.verify_run(run,43,'sbsmi')
            for name in ('a','b'):
                p=Path(tmp)/t.OUTPUT/'seed_43/sbsmi'/name;p.mkdir(parents=True);(p/'COMPLETE.json').write_text('{}')
            with self.assertRaises(ValueError):t.discover(43,'sbsmi')


if __name__=='__main__':unittest.main()

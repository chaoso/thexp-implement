from thexp import Trainer
from thexp.frame import callbacks

from .. import GlobalParams


class __CB(callbacks.TrainCallback):
    pass


class BaseCBMixin(Trainer):
    def callbacks(self, params: GlobalParams):
        from thexp import callbacks
        callbacks.LoggerCallback().hook(self)  # auto log in screen and file
        callbacks.EvalCallback(*params.eval_test_per_epoch).hook(self)  # auto eval/test per 5/10 epoch
        callbacks.AutoRecord().hook(self)  # auto record meter by SummaryWritter

        callbacks.ReportSche().hook(self)
        callbacks.LRSchedule().hook(self)  # auto get params.lr_sche to apply lr rate
        if params.ema:
            callbacks.EMAUpdate().hook(self)  # auto update module named with prefix `ema`

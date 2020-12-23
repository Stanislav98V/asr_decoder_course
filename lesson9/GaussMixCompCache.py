import AcoModels3
import numpy as np

EmptyDistValue=np.float32("inf")

class GaussMixCompCache:
    def __init__(self, AcoModelSet, AcoModel):
        self.AcoModelSet = AcoModelSet
        self.AcoModel = AcoModel
        self.ftr = None
        self.mod = AcoModels3.loadAcoModelSet('out_model2')
        self.cache = np.empty(AcoModels3.AcoModelSet.getUgidCount(self.mod), dtype=np.float32)


    def reset(self):
        self.ftr=None
    pass

    def setFrame(self,ftr):
        self.ftr = ftr
        self.cache.fill(EmptyDistValue)
    pass

    def getDist(self,ugid):
        assert self.ftr is not None
        if self.cache[ugid] == EmptyDistValue:  # True:#
            model = opr_fon(ugid, self.mod)
            self.cache[ugid] = self.AcoModel.dist(model, self.ftr, t=1)
        return self.cache[ugid]
    pass
pass

def opr_fon(ugid, mod):
    count = 0
    for key, model in mod.name2model.items():
        if count == ugid:
            return model
        count += 1
        er = model
    return er




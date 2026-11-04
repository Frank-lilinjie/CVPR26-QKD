def get_model(model_name, args):
    name = model_name.lower()
    if name=='quan_adapter':
        from models.quan_adapter import Learner
    elif name=='quan_prompt':
        from models.quan_prompt import Learner
    elif name=='linear_adapter':
        from models.linear_adapter import Learner
    elif name=='prototype_adapter':
        from models.prototype_adapter import Learner
    else:
        assert 0
    
    return Learner(args)

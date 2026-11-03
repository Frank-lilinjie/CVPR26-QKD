def get_model(model_name, args):
    name = model_name.lower()
    if name=='quan_adapter':
        from models.quan_adapter import Learner
    else:
        assert 0
    
    return Learner(args)

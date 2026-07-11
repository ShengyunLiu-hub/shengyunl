def get_model(model_name, args):
    name = model_name.lower()
    if name == "cllora":
        from models.cllora import Learner
    else:
        raise NotImplementedError("Unknown model {}".format(name))

    return Learner(args)

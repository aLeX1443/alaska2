from tensorflow.python.keras.models import model_from_json


def load_trained_tf_model(
    model_save_dir,
    model_name,
    use_best_val_loss=False,
    print_model_summary=True,
):
    """
    Load the model architecture and weights.

    Parameters
    ---------
    model_save_dir : str
        The path to where the .json model architecture and .h5 model weights
        are stored.
    model_name : str
        The name the model is saved under, i.e., the part before '.h5' or
        '.json'.
    use_best_val_loss : bool
    print_model_summary : bool
    """
    with open(f"{model_save_dir}{model_name}.json", "r") as f:
        model = model_from_json(f.read(), custom_objects={"k": K})
    if use_best_val_loss:
        model_name += "_best_val_loss"
    model.load_weights(model_save_dir + model_name + ".h5")
    if print_model_summary:
        print(model.summary())
    return model

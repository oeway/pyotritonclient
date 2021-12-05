## Instructions for Codex code generation

```
To run models with the BioEngine, we need to first import the libarary with "from pyotritonclient import execute", then use:
results = await execute(
    inputs=[
        image.astype("float32"),
        {"diameter": 30}
    ], # A list of inputs, required different number of inputs for different models, each element can be an numpy array, a dict or a list
    server_url='https://ai.imjoy.io/triton', # the url of the server
    model_name='cellpose-python', # the name of the model
    ) # this execute function exectues the model (specified by model_name) on the server and return results. execute is an async function, it should used with await.

Currently, the server https://ai.imjoy.io/triton supports the following models:
    1. "cellpose-python" for image segmentation with the cellpose model, it requires two inputs, the first one is a image numpy array with 3 dimensions and the data type should be float32, the second one is a dictionary for the parameters, the available parameters is "diameter" for specifiy the object size and "channels" to specify the cell and nuclei channel, the default channels value is [1, 2]. The returned results is a dictionary contains a key named "mask" (results["mask"]) for the mask image. To display the image, use "plt.imshow(mask[0, :, :])". To get region properties and morphological features from the mask, use the "regionprops_table" function from scikit-image (the available property names are the same as the regionprops function but without "mean_intensity").
    2. "stardist" for image segmentation with the stardist model: similar to "cellpose-python", it also requires two inputs and has the same input format, however, the first input should be a single channel image with uint16 type and the second input should be an empty dictionary. The returned results contains a mask with a single channel can be displayed "plt.imshow(mask)".
```

To train cellpose models, here is an example:

```python
from pyotritonclient import SequenceExcutor

def get_example_training_images():
    """Get the example training image"""
    import pickle
    import urllib
    # Download the file from `url` and save it locally under `file_name`:
    urllib.request.urlretrieve(
        "https://github.com/imjoy-team/imjoy-interactive-segmentation/releases/download/v0.1.0/train_samples_4.pkl",
        "train_samples_4.pkl",
    )
    train_samples = pickle.load(open("train_samples_4.pkl", "rb"))
    return [(sample[0].astype("float32"), sample[1].astype("uint16")) for sample in train_samples]


async def train(model_id, train_samples, pretrained_model="cyto", model_token=None, epochs=1, steps=1):
    """
    Train a cellpose model
    Arguments:
        - model_id: the id of the model, should be a integer number greater than 0
        - train_samples: a list of training image and mask pairs
        - pretrained_model: the name of a pretrained model to bootstrap the training data
        - model_token: a token string for protecting the model from overwritting by others
        - epochs: the number of epochs to train
    """
    seq = SequenceExcutor(
        server_url="https://ai.imjoy.io/triton",
        model_name="cellpose-train",
        decode_json=True,
        sequence_id=model_id, 
    )
    for epoch in range(epochs):
        losses = []
        for (image, labels) in train_samples:
            inputs = [
                image.astype("float32"),
                labels.astype("uint16"),
                {
                    "steps": steps,
                    "pretrained_model": pretrained_model,
                    "resume": True,
                    "model_token": model_token,
                    "channels": [1, 2],
                    "diam_mean": 30,
                },
            ]
            result = await seq.step(inputs, select_outputs=["info"])
            losses.append(result["info"][0]["loss"])
        avg_loss = np.array(losses).mean()
        print(f"Epoch {epoch}  loss={avg_loss}")

    result = await seq.end(
        decode_json=True,
        select_outputs=["model", "info"],
    )
    # Save the weights
    model_package = result["model"][0]
    filename = result["info"][0]["model_files"][0]
    with open(filename, "wb") as fil:
        fil.write(model_package)
    print(f"Model package saved to {filename}")
```

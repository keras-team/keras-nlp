import keras
from keras import ops

from keras_hub.src.bounding_box.iou import compute_ciou


class CIoULoss(keras.losses.Loss):
    """Implements the Complete IoU (CIoU) Loss

    CIoU loss is an extension of GIoU loss, which further improves the IoU
    optimization for object detection. CIoU loss not only penalizes the
    bounding box coordinates but also considers the aspect ratio and center
    distance of the boxes. The length of the last dimension should be 4 to
    represent the bounding boxes.

    Args:
        bounding_box_format: a case-insensitive string (for example, "xyxy").
            Each bounding box is defined by these 4 values. For detailed
            information on the supported formats, see the [KerasCV bounding box
            documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        epsilon: A small value added to avoid division by zero and stabilize
            calculations.

    References:
        - [CIoU paper](https://arxiv.org/pdf/2005.03572.pdf)

    Example:
    ```python
    y_true = np.random.uniform(
        size=(5, 10, 5),
        low=0,
        high=10)
    y_pred = np.random.uniform(
        (5, 10, 4),
        low=0,
        high=10)
    loss = keras_hub.src.models.yolo_v8.ciou_loss.CIoULoss("xyxy")
    loss(y_true, y_pred).numpy()
    ```

    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=CIoULoss())
    ```
    """

    def __init__(self, bounding_box_format, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.bounding_box_format = bounding_box_format

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)

        if y_pred.shape[-1] != 4:
            raise ValueError(
                "CIoULoss expects y_pred.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_pred.shape[-1]={y_pred.shape[-1]}."
            )

        if y_true.shape[-1] != 4:
            raise ValueError(
                "CIoULoss expects y_true.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_true.shape[-1]={y_true.shape[-1]}."
            )

        if y_true.shape[-2] != y_pred.shape[-2]:
            raise ValueError(
                "CIoULoss expects number of boxes in y_pred to be equal to the "
                "number of boxes in y_true. Received number of boxes in "
                f"y_true={y_true.shape[-2]} and number of boxes in "
                f"y_pred={y_pred.shape[-2]}."
            )

        ciou = compute_ciou(y_true, y_pred, self.bounding_box_format)
        return 1 - ciou

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
            }
        )
        return config

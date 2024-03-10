
import cv2
import einops
import numpy as np
import torchvision.transforms as tfm
from PIL import Image, ImageDraw, ImageFont


# Height and width of a single image
H = 512
W = 512
TEXT_H = 175
FONTSIZE = 80
SPACE = 50  # Space between two images
K_VIZ = 5  # Num of predictions to show in visualizations


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with vertical text, spaced along rows."""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
    img = Image.new('RGB', ((W * len(labels)) + 50 * (len(labels)-1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        _, _, w, h = d.textbbox((0,0), text, font=font)
        d.text(((W+SPACE)*i + W//2 - w//2, 1), text, fill=(0, 0, 0), font=font)
    return np.array(img)


def draw(img, c=(0, 255, 0), thickness=20):
    """Draw a colored (usually red or green) box around an image."""
    p = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    for i in range(3):
        cv2.line(img, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), c, thickness=thickness*2)
    return cv2.line(img, (p[3, 0], p[3, 1]), (p[0, 0], p[0, 1]), c, thickness=thickness*2)


def create_predictions_image(images_paths_angle_correct):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    """
    pad = SPACE // 2
    images = []
    labels = []
    for image_path, angle, is_correct in images_paths_angle_correct:
        if is_correct is None:
            label = "Query"
        else:
            label = f"Pred - {is_correct}"
        labels.append(label)
        pil_image = Image.open(image_path)
        pil_image = tfm.Resize([H, W])(pil_image)
        rotated_image = np.array(tfm.functional.rotate(pil_image, angle=angle))
        
        if is_correct is not None:
            # Draw red or green square around each prediction
            color = (0, 255, 0) if is_correct else (255, 0, 0)
            draw(rotated_image, color)
        
        padded_rotated_image = np.pad(rotated_image, [[0, 0], [pad, pad], [0, 0]], constant_values=255)
        images.append(padded_rotated_image)
    concat_image = einops.rearrange(images, "n h w c -> h (n w) c")
    concat_image = concat_image[:, pad : -pad]
    try:
        labels_image = write_labels_to_image(labels) * 255
        final_image = np.concatenate([labels_image, concat_image])
    except OSError:  # Handle error in case of missing PIL ImageFont
        final_image = concat_image
    final_image = Image.fromarray(final_image)
    return final_image


def compute_visualizations(eval_ds, log_dir, num_preds_to_save, predictions, preds_angles, positives_per_query):
    
    viz_log_dir = log_dir / f"preds_{eval_ds.dataset_name}"
    viz_log_dir.mkdir()
    for num_q in range(0, eval_ds.num_q, eval_ds.num_q // num_preds_to_save):
        
        images_paths_angle_correct = [(eval_ds.queries_paths[num_q], 0, None)]
        
        # Append the first K_VIZ preds
        for pred, angle in zip(predictions[num_q, :K_VIZ], preds_angles[num_q, :K_VIZ]):
            is_correct_pred = pred in positives_per_query[num_q]
            images_paths_angle_correct.append((eval_ds.db_paths[pred], int(angle), is_correct_pred))
        
        pred_image = create_predictions_image(images_paths_angle_correct)
        pred_image.save(viz_log_dir / f"{num_q:04d}.jpg")


import numpy as np
import matplotlib.pyplot as plt

def plot_image_grid(
  images,
  filepath,
  titles=None,
  fontsize=None,
  plot_grid_indices=False,
  grid_positions=None,
  size_scaling_factor=1.0
):
  if grid_positions is None:
    grid_positions = np.array([
      [0, i] for i in range(len(images))
    ])
  num_rows = np.max(grid_positions[:, 0]) + 1
  num_cols = np.max(grid_positions[:, 1]) + 1

  # Determine the maximum width and height among all images
  max_height = max(image.shape[0] for image in images)
  max_width = max(image.shape[1] for image in images)

  base_dpi = 100 # Arbitrary value (except for font), as the height and width are set accordingly.
  fig_height = (max_height * num_rows) / base_dpi
  fig_width = (max_width * num_cols) / base_dpi
  fig, axs = plt.subplots(
    num_rows,
    num_cols,
    figsize=(fig_width, fig_height),
    dpi=base_dpi * size_scaling_factor
  )

  for i in range(len(images)):
    row, col = grid_positions[i]
    # Display each image in its respective subplot
    image = images[i]
    if num_rows == 1 and num_cols == 1:
      axis = axs
    elif num_rows == 1:
      axis = axs[col]
    elif num_cols == 1:
      axis = axs[row]
    else:
      axis = axs[row, col]
    axis.imshow(image)
    axis.axis("off")
    if titles is not None:
      axis.set_title(titles[i], fontsize=fontsize)
    if plot_grid_indices:
      axis.text(
        x=0,
        y=0,
        s=f"{i}",
        color='black',
        fontsize=fontsize,
        ha='left',
        va='top',
        backgroundcolor='white'
      )

  # Adjust subplot parameters to reduce space
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)

  fig.savefig(
    filepath,
    bbox_inches="tight",
    pad_inches=0,
  )
  plt.close()
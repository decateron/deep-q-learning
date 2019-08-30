from skimage.color import rgb2gray
from skimage import transform
import numpy as np



def preprocess_frame(frame, crop_size):
    """
    Grayscale each of our frames. Crop the screen. Normalize pixel values. Resize the preprocessed frame.

    :param frame: Just a frame.
    :return: preprocessed_frame
    """

    # Grayscale frame
    gray = rgb2gray(frame)

    # Crop the screen
    # [Up: Down, Left: Right]
    cropped_frame = gray[crop_size[0]: crop_size[1], crop_size[3]: crop_size[4]]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame   # 110x84x1 frame



def stack_frames(state, is_new_episode, stack_size=4):
    """
    Preprocess frame. Append the frame to the deque. Build the stacked state

    :param state: The actual state in the game.
    :param is_new_episode: "True" means that new episode starts. False means that old episode continues.
    :param stack_size: How many states we want to stack ?
    :return: stacked_state and current stacked frames.
    """

    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames(initialize deque with zero-images one array for each image)
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range (stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4 times
        for i in range(4):
            stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames




"""
Module:
    gif.py
Overwiew:
    This module can be used to generate gifs from frames.
Functions:
    generate_gif
"""



from skimage import transform
import imageio
import numpy as np



def generate_gif(frame_number, frames_for_gif, reward, path):
    """
    Takes a sequence of frames and then create a gif from them.

    :param frame_number: The number of the frame.
    :param frames_for_gif: The sequence of frames, which represent one episode.
    :param reward: The reward that the agent has received for an episode.
    :param path: Path to save our gif
    :return: pass
    """

    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = transform.resize(frame_idx, (420, 320, 3),
                                        preserve_range=True, order=0).astype(np.uint8)
    
    imageio.mimsave(f'{path}{"_ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1/30)
                    

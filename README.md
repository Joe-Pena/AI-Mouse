MouseAI, an Artificially Intelligent Mouse!

## Intro

This is a simple project to immerse one's self into the world of Artificial Intelligence. It's meant as a simple intro into the subject matter.

### Requirements

In order to be able to interact with this project I recommend to install [https://www.anaconda.com/download/](Anaconda), which is a python platform which proves itself very useful in scenarios where machine learning, deep learning, and other data science subjects are being used.

The following packages are to be installed for the program to run smoothly, note the versions on each of the packages, as there are newer versions available for each, but this program requires these particular versions in order to run correctly. You can install the packages by using `conda install [SOURCE] [PACKAGE NAME]`

* Python 3.5
* Pytorch-cpu 0.3.1
* Keras 2.2.2
* Tensorflow 1.3.0
* Kivy 1.10.1
* matplotlib 2.0.2

## How to use

I recommend using the Spyder IDE included with the Anaconda distribution, it includes a text editor, a file explorer, and an IPython console, which make running the program easier.

In order to run it, make sure the spyder file explorer is opened and in the same directory as where the `map.py` and `ai.py` are located.

Afterwards simply right click on `map.py` in the file explorer and click on the green `run` button. This will open up a window with a black background and our mouse (A white rectangle) simply running around the window.

And that's it, the program is running!

To make it more fun, simply draw a maze in the window where the mouse is running, you will notice that you're able to draw yellow lines, our mouse hates these lines, so he will try to avoid them! After going through a few times though, of course.

You can watch the mouse slowly learn the layout of the new maze, as it will try to get to it's two goals, the top left corner and the bottom right corner of the window. Once you feel like your newly created mouse is intelligent enough, simply click on the `save` button to save the brain of the mouse as a file named `last_brain.pth`, when you start a new game, simply click the `load` button to reload the neural network inside of `last_brain.pth`.

Last thing, if you get out of hand with the yellow color, or simply want to draw a different path for our mouse to try then click the `clear` button to clear the entire window so you can draw something new!
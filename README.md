# 2D-Image-Convultion-Cuda


Compile the code with:

```
nvcc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs convergence.cu  
```

If you want to dowload openCV modules follow the [tutorial from the official website](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html).

And if the program doesn't recognize the library opencv2 try with:

```
ln -s /usr/local/include/opencv4/opencv2/ /usr/local/include/opencv2
```


For some reason if you increase the dimentions of the image, the program just dies.

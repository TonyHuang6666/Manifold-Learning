# ***Face-Recognition-using-LDA***
Performing face recognition with complete implementation in python of LDA algorithm for the classification of the input labeled images


## **Data Set**
I used the **ORL dataset** which has 10 images per 40 people,Every image is a grayscale image of size **92x112**.<br/>
For more information about the data set:<br/>
https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html



## **Steps**
1. Read the images and convert into a vector of **10304 (92*112)** values corresponding to the image size.
2. Split the data to training and testing with a percantage of **50 %** for each batch
3. Apply the **LDA algorithm** steps with a final goal of computing the **eigen-values** and **eigen-vectors**
4. Apply the **KNN algorithm** with different K valuesfor the calssification phase, predicting the values of the test data batch and calculating the **accuracy**<br/>


## **LDA Algorithm**
1. Compute a **mean matrix** of size **(40,10304)** whre each row maps to the mean vector for a class
2. Compute an overall mean for all classes producing a vector of size **(10304,1)**
3. Compute the **between class scater matrix** producing a large matrix of size **(10304,10304)**
![Capture](https://user-images.githubusercontent.com/37254194/65734405-11087900-e0d3-11e9-9e5c-a9f72d23dd2b.PNG)
4. Compute the **center class scatter matrix**
5. Compute the **within class scatter matrix**
6. Compute the **W** value 
7. Finally compute the **eigen values and eigen vectors** limited to the number of classes **(40)**<br/>



## **Applying KNN**
- Before applying the KNN algorithm I mapped the training and testing data batches to the **new dimensions** by the **dot product of each image matrix and the eigen vectors matrix.** 
- Applying KNN with different K values ranging from **(1 -> 25)** to get the estimated **best value for K**.


## **Results**
- The algorithm performed in an acceptable way giving a best recognition accuracy of **95.5 %**.
![Capture](https://user-images.githubusercontent.com/37254194/65734760-cb4cb000-e0d4-11e9-954f-0de6e8375431.PNG)<br/>

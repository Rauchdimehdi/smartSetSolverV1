# smartSetSolverV1


Set is a real-time card game designed by Marsha Falco in 1974 and published by Set Enterprises in 1991. The deck consists of 81 unique cards that vary in four features across three possibilities for each kind of feature: number of shapes (one, two, or three), shape (diamond, squiggle, oval), shading (solid, striped, or open), and color (red, green, or purple)

![DEmonstrateur](https://user-images.githubusercontent.com/40724965/75486834-90b1b100-59ad-11ea-80a9-bfb38634d217.png)

> Our application is composed mainly from a Raspberry & Google Coral ( Edge Computing )

![Screenshot from 2020-01-27 15-01-55](https://user-images.githubusercontent.com/40724965/73926564-22a72c00-48d0-11ea-9e3c-3bb4ba48dd82.png)


+Step 1: To be able to detect Set cards we need a model that is pre-trained to do that. That's why we started by creating our own model that recognize set cards cards using ' Transfer Learning '. For more information, please open google colab file named 0_3CopiedeSetSolverHub_V3_1.ipynb to understand how we did that.
After creating a keras model that can finally detect which cards, we converted it to Tensorflow Lite in order to run it in Google Coral attached to Raspberry.


+Step 2: In order to apply our model, we need to isolate apart each card that is posed on the desk. Opencv was the best solution for that. We create a filter that return the borders of all the cards ( white area)

![1](https://user-images.githubusercontent.com/40724965/73926958-d3adc680-48d0-11ea-84ea-961741b2cbe2.png)

+Step 3: Apllying the Tensorflow lite model to detect cards

![modelApplied](https://user-images.githubusercontent.com/40724965/75487078-0ae23580-59ae-11ea-89d3-fce3ae96cdeb.png)

![Screenshot from 2020-01-27 15-01-55](https://user-images.githubusercontent.com/40724965/73927021-efb16800-48d0-11ea-95aa-22574cd285ed.png)

+Step 4: Excute our algorithme to check the set
![Screenshot from 2020-02-04 11-50-18](https://user-images.githubusercontent.com/40724965/73927107-0fe12700-48d1-11ea-9d35-a2ea39f89f5b.png)
![Screenshot from 2020-02-04 11-53-00](https://user-images.githubusercontent.com/40724965/73927110-11125400-48d1-11ea-8626-662f65a25d42.png)



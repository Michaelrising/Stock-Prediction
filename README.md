# Stock-Prediction

## In this project, I implemented two methods to predict the stock returns given the attributes (90 features in total)



#### Here is the simple illustration of the MLP auto encoder decoder model. Since the input and output are noisy with a low informtion-noise ratio. In first use a Gaussion Noise layer to prevent overfitting and apply dropout layers in the encoder and decoder. 

#### Since our goal is maximzing the correlation between the predcition and the groud truth stock returns. I apply the pearson correlation as part od the training loss, but not as a standalone training loss. Because using only the correlation as the loss may make the training unstable and the results may be totally bad. So We use MSE(y_true, y_pred) + $\lambda$ 1/IC(y_true, y_pred) as the training loss. In my setting, I set $\lambda = 0.05$. For mlp decoder, we apply MSE(x_pred, x_true) as the training loss. 

### MLP-autoencoder decoder
![mlp_autoencoder_decoder](https://user-images.githubusercontent.com/53537769/230900736-2d978fd5-1c5f-45ea-bb99-8e434c1591b5.jpeg)

### Mega Model
![mega_model](https://user-images.githubusercontent.com/53537769/230901866-e5e07b04-88b3-4445-b1b0-0c2b3ce5ec22.jpeg)

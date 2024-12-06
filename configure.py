def get_default_config(data_name):
    if data_name in ['MNIST-USPS']:
        return dict(
            Autoencoder=dict(
                arch0=[784, 1024, 1024, 1024, 128],
                arch1=[784, 1024, 1024, 1024, 128],
                activations0='relu',
                activations1='relu',
                batchnorm=True,
            ),
            training=dict(
                pre_epochs = 0,
                con_epochs = 150,
                temperature = 1,
                view = 2,
                missing_rate=0.3,
                seed=7,
                batch_size=256,
                epoch=400,
                lr=1.0e-4,
                lambda1=1,
                lambda2=1,
                lambda3=0.01,
                n_class=10,
            ),
        )
    elif data_name in ['BDGP']:
        return dict(
            Autoencoder=dict(
                arch0=[79, 1024, 1024, 1024, 128],
                arch1=[1750, 1024, 1024, 1024, 128],
                activations0='relu',
                activations1='relu',
                batchnorm=True,
            ),
            training=dict(
                pre_epochs = 50,
                con_epochs = 150,
                temperature = 0.5,
                view=2,
                seed=6,
                missing_rate=0.3,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=10,
                lambda2=10,
                lambda3=0.1,
                n_class=5,
            ),
        )
    elif data_name in ['Scene-15']:
        return dict(
            Autoencoder=dict(
                arch0=[59, 1024, 1024, 1024, 128],
                arch1=[20, 1024, 1024, 1024, 128],
                activations0='relu',
                activations1='relu',
                batchnorm=True,
            ),
            training=dict(
                pre_epochs = 0,
                con_epochs = 100,
                temperature=1,
                view = 2,
                missing_rate=0.3,
                seed=19,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                lambda1=0.01,
                lambda2=0.1,
                lambda3=0.1,
                n_class=15,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
        )
    elif data_name in ['LandUse-21']:
        return dict(
            Autoencoder=dict(
                arch0=[59, 1024, 1024, 1024, 128],
                arch1=[20, 1024, 1024, 1024, 128],
                activations0='relu',
                activations1='relu',
                batchnorm=True,
            ),
            training=dict(
                pre_epochs = 50,
                con_epochs = 50,
                temperature = 1,
                view = 2,
                missing_rate=0.3,
                seed=7,
                batch_size=512,
                epoch=500,
                lr=1.0e-4,
                lambda1=100,
                lambda2=0.1,
                lambda3=0.01,
                n_class=21,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
        )
    elif data_name in ['NoisyMNIST']:
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[784, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                view = 2,
                missing_rate=0.3,
                seed=2,
                batch_size=128,
                epoch=50,
                lr=1.0e-4,
                lambda1=10.0,
                lambda2=0.1,
                n_class=10,
            ),
        )
    elif data_name in ['HandWritten']:
        return dict(
            Autoencoder=dict(
                arch0=[240, 1024, 1024, 1024, 128],
                arch1=[76, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                arch3=[47, 1024, 1024, 1024, 128],
                arch4=[64, 1024, 1024, 1024, 128],
                arch5=[6, 1024, 1024, 1024, 128],
                activations0='relu',
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                pre_epochs = 0,
                con_epochs = 15,
                temperature = 1,
                view = 6,
                missing_rate=0.3,
                seed=2,
                batch_size=128,
                epoch=50,
                lr=1.0e-4,
                lambda1=10.0,
                lambda2=0.1,
                lambda3=0.1,
                n_class=10,
            ),
        )
    elif data_name in ['Caltech101-7']:
        return dict(
            Autoencoder=dict(
                arch0=[48, 1024, 1024, 1024, 128],
                arch1=[40, 1024, 1024, 1024, 128],
                arch2=[254, 1024, 1024, 1024, 128],
                arch3=[1984, 1024, 1024, 1024, 128],
                arch4=[512, 1024, 1024, 1024, 128],
                arch5=[928, 1024, 1024, 1024, 128],
                activations0='relu',
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                temperature = 1,
                view = 6,
                missing_rate=0.3,
                seed=2,
                batch_size=128,
                epoch=50,
                lr=1.0e-4,
                lambda1=10.0,
                lambda2=0.1,
                lambda3=0.1,
                n_class=10,
            ),
        )
    else:
        raise Exception('Undefined data_name')

from face_template_gan import FaceTemplateGAN

if __name__ == '__main__':
    ftm = FaceTemplateGAN()
    ftm.train(train_gen=True, train_disc=True)
    # ftm.test()

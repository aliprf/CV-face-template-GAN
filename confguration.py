class Config:
    # _prefix_path = '/media/data3/ali/FL/new_data/300W/'  # --> zeus
    # _prefix_path = '/media/data2/alip/FL/new_data/300W/'  # --> atlas
    _prefix_path = '/media/ali/data/new_data/300W/'  # --> local

    annotation_path = _prefix_path + 'training_set/no_aug/annotations/'

    image_input_size = 256
    net_image_input_size = 256
    noise_input_size = 500

    # batch_size = 2
    batch_size = 500
    epochs = 50000

    num_of_landmarks = 136

    flatten_img_size = 28*28
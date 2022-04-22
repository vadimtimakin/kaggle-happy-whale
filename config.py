from omegaconf import OmegaConf

config = {
    'seed': 0xFACED,

    'num_classes': 15587,

    'training': {
        'num_epochs': 15,
        'early_stopping': 5,
        'device': 'cuda',

        'mixed_precision': False,
        'gradient_accumulation': False,
        'gradient_clipping': False,
        'freeze_batchnorms': False,
        'gradient_accumulation_steps': 4,
        'clip_value': 2,
        
        'debug': False,
        'number_of_debug_train_samples': 100,
        'number_of_debug_val_samples': 100,

        'n_neighbors': 100,
    },

    'paths': {
        'path_to_images': '/home/toefl/datasets/datasets/backfins/train_images',
        'path_to_csv': '/home/toefl/datasets/datasets/happy-whale-and-dolphin/train.csv',
        'path_to_masks': '/home/toefl/datasets/datasets/masks/train',
        'path_to_sample_sub': '/home/toefl/datasets/datasets/happy-whale-and-dolphin/sample_submission.csv',
        'path_to_test': '/home/toefl/datasets/datasets/backfins/test_images',
        'save_dir': '/home/toefl/K/dolphin/checkpoints/convnext_large_nomp',
        'path_to_checkpoint': None,
        'path_to_pretrain': None
    },

    'data': {
        'id_column': 'image',
        'target_column': 'individual_id',
        'split_column': 'species_and_ids',
        'n_folds': 5,

        'dataloader_params': {
            'batch_size': 2,
            'num_workers': 8,
            'pin_memory': False,
            'persistent_workers': True,
            'drop_last': True,
        },
    },

    'model_params': {
        'backbone':'tf_efficientnet_b7',
        'embedding_size': 512,
        'dropout': 0.2,
        'n_classes': '${num_classes}',
    },

    'criterion_params': {
        's': 45,
        'm': 0.4,
        'crit': 'bce',
        'class_weights_norm': 'batch',
    },

    'logging': {
        'log': True,
        'wandb_username': 'toefl',
        'wandb_project_name': 'dolphin'
    },

    'inference_checkpoints': [
        '/home/toefl/K/dolphin/checkpoints/flips/fold_0/best.ckpt',
    ]
}

config = OmegaConf.create(config)

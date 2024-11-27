# ACRL hyperparameters configuration

config = {
    # --- CURRICULUM ---
    'num_encoder_update': 5,
    'noise_std': [0.5, 0.5],
    'update_freq': 10, # episode
    'warmup_step': 10000,

    # --- VAE TRAINING ---
    # general
    'lr_vqvae': 0.005,
    'vq_batch_size': 256,
    'vq_buffer_size': 1e4,
    'pretrain_len': 0,
    'vq_beta': 0.25,
    'codebook_k': 64,
    'latent_dim': 64,

    # - encoder
    'state_embedding_size': 128,
    'encoder_layers': [128, 128],

    # - decoder
    'decoder_layers': [128, 128, 128],
}

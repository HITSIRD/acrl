# ACRL hyperparameters configuration

config = {
    # --- CURRICULUM ---
    'step_size': 0.9,
    'return_delta': 0.4,  # select traj sample which return is greater than return_delta
    'update_delta': 0.5,  # if mean of return > update_delta, then update context dist
    'target_return_threshold': 0.1,
    'lambda': 0.9,
    'task_buffer_size': 256,
    'noise_std': [0.5, 0.5, 0.5, 0.5],
    'num_target_samples': 20,
    'update_freq': 500,
    'enable_lsp': True,
    'enable_ebu': True,
    'encoder_max_grad_norm': None,
    'decoder_max_grad_norm': None,
    # --- VAE TRAINING ---
    # general
    'lr_vae': 0.005,
    'lr_task_decoder': 0.005,
    'size_vae_buffer': 2048,
    'vae_buffer_add_thresh': 1,
    'vae_batch_num_trajectories': 16,
    'vae_avg_reconstruction_terms': False,
    'num_vae_update': 2,
    'num_encoder_update': 100,
    'max_task_decoder_update': 8,
    'task_decoder_loss_threshold': 0.001,
    'pretrain_len': 0,
    'kl_weight': 1.0,

    # - transition_encoder
    'action_embedding_size': 8,
    'state_embedding_size': 64,
    'reward_embedding_size': 8,
    'task_embedding_size': 64,
    'encoder_layers': [128, 128],
    'latent_dim': 4,

    # - decoder: rewards
    'decode_reward': True,
    'rew_loss_coeff': 1.0,
    'reward_input_prev_state': False,
    'reward_input_action': False,
    'reward_input_next_state': False,
    'reward_decoder_layers': [64, 64],
    'multihead_for_reward': False,
    'rew_pred_type': 'bernoulli',

    # - decoder: state transitions
    'decode_state': True,
    'state_loss_coeff': 0.01,
    'state_input_prev_state': False,
    'state_input_action': False,
    'state_decoder_layers': [128, 128],
    'state_pred_type': 'deterministic',

    # - decoder: task
    'decode_task': False,
    'task_loss_coeff': 1.0,
    'task_decoder_layers': [128, 128],
    'task_pred_type': 'param',
    'task_batch_num_trajectories': 128,

    # --- ABLATIONS ---
    # for the VAE
    'disable_decoder': False,
    'disable_stochasticity_in_latent': False,
    'disable_kl_term': False,
    'decode_only_past': False,
    'kl_to_gauss_prior': True,

    # combining vae and RL loss
    'rlloss_through_encoder': False,
    'add_nonlinearity_to_latent': False,
    'vae_loss_coeff': 0.01,

    # for the policy training
    'sample_embeddings': False,

    # general settings
    'deterministic_execution': False
}

import os

class FileManager:
    def __init__(self, args):
        self.args = args
    
    def pretrained_filename_base(self):
        return (
            "bs_"
            + str(self.args.warmup_batch_size)
            + "_lr_"
            + str(self.args.warmup_lr)
            + "_seed_"
            + str(self.args.seed)
            + "_rate_"
            + str(self.args.server_rate)
            + "_maxrate_"
            + str(self.args.max_server_rate)
            + "_clients"
            + str(self.args.num_users)
        )

    def pretrained_filename(self):
        return (
            "bs_"
            + str(self.args.warmup_batch_size)
            + "_lr_"
            + str(self.args.warmup_lr)
            + "_seed_"
            + str(self.args.seed)
            + "_rate_"
            + str(self.args.server_rate)
            + "_maxrate_"
            + str(self.args.max_server_rate)
            + "_clients"
            + str(self.args.num_users)
            + "_warmup_epoch_"
            + str(self.args.eval_from)
        )
    
    def train_filename(self):
        return (
            "nclients_"
            + str(self.args.num_users)
            + "_warmupbs_"
            + str(self.args.warmup_batch_size)
            + "_bs_"
            + str(self.args.batch_size)
            + "_warmuplr_"
            + str(self.args.warmup_lr)
            + "_lr_"
            + str(self.args.lr)
            + "_iters_"
            + str(self.args.train_epochs)
            + "_seed_"
            + str(self.args.seed)
            + "_serverrate_"
            + str(self.args.server_rate)
            + "_maxrate_"
            + str(self.args.max_server_rate)
            + "_mode_"
            + str(self.args.mode)
            + "_iid_"
            + str(self.args.iid)
            + "_evalfrom_"
            + str(self.args.eval_from)
            + "_mu_"
            +str(self.args.mu)
        )

    def remove_filename(self):
        return (
            self.train_filename()
            + "_removedclients_"
            + "_".join(map(str, self.args.remove_idx))
            + "_epochs_"
            + str(self.args.num_epochs)
            + "removal_lr"
            + str(self.args.removal_lr)
            + "weightdecay"
            + str(self.args.weight_decay)
        )
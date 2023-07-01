import os
config_template = '''
[task]
task_name = GRU-h{hidden_size}-l{num_layer}-ini_{init_weight}

[model]
name = RNN_GRU
hidden_size = {hidden_size}
num_layers = {num_layer}
drop_keep_prob = 0.3
update_w2v = True
init_weight = {init_weight}

[train]
batch_size = 16
num_epochs = 20
lr = 1e-3
max_length = 120
'''

hidden_sizes = [64, 128, 256, 512]
num_layers = [1, 2, 4]
init_weights = ["None","kaiming","xavier"]

if not os.path.exists("GRU"):
    os.makedirs("GRU")
# 创建子文件夹
for hidden_size in hidden_sizes:
    for num_layer in num_layers:
        for init_weight in init_weights:
            config_str = config_template.format(hidden_size=hidden_size, num_layer=num_layer,init_weight=init_weight)
            with open(f"GRU/GRU-h{hidden_size}-l{num_layer}-ini_{init_weight}.cfg", "w") as f:
                f.write(config_str)
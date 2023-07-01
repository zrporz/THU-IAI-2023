import os
config_template = '''
[task]
task_name = MLP-h[{hidden_dim}]-ini_{init_weight}

[model]
name = MLP
hidden_dim = {hidden_dim}
drop_keep_prob = 0.3
update_w2v = True
init_weight = {init_weight}

[train]
batch_size = 20
num_epochs = 10
lr = 1e-3
max_length = 120
'''

hidden_dims = [[512,256,128,64],[512,256,128,64,32],[512,256,128,64,32,16]]
init_weights = ["None","kaiming","xavier"]

if not os.path.exists("MLP"):
    os.makedirs("MLP")
# 创建子文件夹
for hidden_dim in hidden_dims:
    for init_weight in init_weights:
        config_str = config_template.format(hidden_dim=','.join(str(x) for x in hidden_dim),init_weight=init_weight)
        with open(f"MLP/MLP-h{hidden_dim}-ini_{init_weight}.cfg", "w") as f:
            f.write(config_str)
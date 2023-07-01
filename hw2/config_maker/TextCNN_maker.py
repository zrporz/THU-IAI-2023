import os
config_template = '''
[task]
task_name = TextCNN-ks[{kernel_size}]-kn{kernel_num}-ini_{init_weight}

[model]
name = TextCNN
kernel_num = {kernel_num}
kernel_size = {kernel_size}
drop_keep_prob = 0.3
update_w2v = True
init_weight = {init_weight}

[train]
batch_size = 16
num_epochs = 20
lr = 1e-3
max_length = 120
'''

kernel_sizes = [[3,5],[3,5,7],[3,5,7,9]]
kernel_nums = [2,8,14,20]
init_weights = ["None","kaiming","xavier"]

if not os.path.exists("TextCNN"):
    os.makedirs("TextCNN")
# 创建子文件夹
for kernel_size in kernel_sizes:
    for kernel_num in kernel_nums:
        for init_weight in init_weights:
            config_str = config_template.format(kernel_num=kernel_num, kernel_size=','.join(str(x) for x in kernel_size),init_weight=init_weight)
            with open(f"TextCNN/TextCNN-ks{kernel_size}-kn{kernel_num}-ini_{init_weight}.cfg", "w") as f:
                f.write(config_str)
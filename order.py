import os

orders = []
with open('order/order_SMP_train.txt', encoding='utf-8') as f1, open('order/order_SMP_test.txt', encoding='utf-8') as f2:
#with open('order/order_COAE_train.txt', encoding='utf-8') as f1, open('order/order_COAE_test.txt', encoding='utf-8') as f2:
# with open('order/order_SEMEVAL_train.txt', encoding='utf-8') as f1, open('order/order_SEMEVAL_test.txt', encoding='utf-8') as f2:
    for line in f1: # orders列表拼接俩文件中的内容
        line = line.replace('\ufeff','').strip()
        if line != '' and '#' not in line:
            orders.append(line)
    for line in f2:
        line = line.replace('\ufeff','').strip()
        if line != '' and '#' not in line:
            orders.append(line)

for j in range(len(orders)):
    os.system(orders[j]) # os.system(cmd)即可在python中使用linux命令。
    # os.system(command) command  要执行的命令，相当于在Windows的cmd窗口中输入的命令。如果要向程序或者
    # 脚本传递参数，可以使用空格分隔程序及多个参数。
    print(orders[j],'\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

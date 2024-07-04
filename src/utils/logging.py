import coloredlogs

coloredlogs.install()

def to_string(tensor):
    lists = [str(i) for i in tensor.to_list()]
    return ', '.join(lists)
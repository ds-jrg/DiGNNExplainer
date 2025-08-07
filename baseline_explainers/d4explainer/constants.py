feature_dict = {
    "BA_shapes": 4,
    "Tree_Cycle": 2,
    "Tree_Grids": 2,
    "mutag": 7,
    "ba3": 4,
    "dblp": 4,
    "imdb": 3
}

task_type = {
    "BA_shapes": "nc",
    "Tree_Cycle": "nc",
    "Tree_Grids": "nc",
    "mutag": "gc",
    "ba3": "gc",
    "dblp": "nc",
    "imdb": "nc"
}

dataset_choices = list(task_type.keys())

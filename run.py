from GGANet.RunModel import *

if __name__ == "__main__":
    exp_name = "Normal"
    hp = config()
    for dataset in  ["DrugBank", "Enzyme", "GPCRs", "ion_channel"]:
        trainer = Trainer(GeoDTI, hp, exp_name, dataset,watch_time=False)
        trainer.run()
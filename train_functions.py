from torch import embedding
from torch.cuda import amp
import time
import gc
import wandb
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import math 
import matplotlib.pyplot as plt

from utils import *
from objects.model import LesGoNet
from objects.optimizer import MADGRAD
from objects.scheduler import GradualWarmupSchedulerV2
from objects.loss_function import ArcFaceLoss, ArcFaceLossAdaptiveMargin
from objects.dolg_model import DOLG
from data import get_fold_margins, get_loaders


def extract_embeddings(config, model, dataloader):
    embeddings = []
    targets = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            inputs, labels = batch
            inputs = inputs.squeeze().to(config.training.device)

            outputs = model(inputs.float(), get_embeddings=True)
            embeddings.append(outputs.cpu())
            targets += labels

    return np.concatenate(embeddings), np.array(targets)


def train_loop(config, model, train_loader, optimizer, loss_function, scaler):
    '''Train loop.'''

    print('Training')

    model.train()

    if config.training.freeze_batchnorms:
        for name, child in (model.named_children()):
            if name.find('BatchNorm') != -1:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
    
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch
        inputs, labels = inputs.squeeze().to(config.training.device), labels.to(config.training.device)

        if not config.training.gradient_accumulation:
            optimizer.zero_grad()
        
        if config.training.mixed_precision:
            with amp.autocast():
                outputs = model(inputs.float())

                loss = loss_function(outputs, labels)

                if config.training.gradient_accumulation:
                    loss = loss / config.training.gradient_accumulation_steps
        else:
            outputs = model(inputs.float())

            loss = loss_function(outputs, labels)

        total_loss += loss.item()

        if config.training.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if config.training.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)

        if config.training.gradient_accumulation:
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        elif config.training.mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    return total_loss / len(train_loader)


def val_loop(config, model, val_loader, train_loader):
    '''Validation loop.'''

    new_individual_thres = 0.5

    print('Validating')

    model.eval()

    print("Extracting train embeddings")
    train_embeddings, train_labels = extract_embeddings(config, model, train_loader)

    print("Extracting val embeddings")
    val_embeddings, val_labels = extract_embeddings(config, model, val_loader)

    neigh = NearestNeighbors(n_neighbors=config.training.n_neighbors,metric="cosine")
    neigh.fit(train_embeddings)

    distances,idxs = neigh.kneighbors(val_embeddings, return_distance=True)
    conf = 1-distances
    preds=[]

    for j in range(len(idxs)):
        preds.append(list(train_labels[idxs[j]]))

    allTop5Preds=[]
    valid_labels_list=[]
    for i in range(len(preds)):
        valid_labels_list.append((val_labels[i]))

        predictTop = preds[i][:5]
        Top5Conf = conf[i][:5]

        if Top5Conf[0] < new_individual_thres:
           
            tempList=['new_individual',predictTop[0],predictTop[1],predictTop[2],predictTop[3]]
            allTop5Preds.append(tempList)   
           
        elif Top5Conf[1] < new_individual_thres:
   
            tempList=[predictTop[0],'new_individual',predictTop[1],predictTop[2],predictTop[3]]
            allTop5Preds.append(tempList)    
           
        elif Top5Conf[2] < new_individual_thres:

            tempList=[predictTop[0],predictTop[1],'new_individual',predictTop[2],predictTop[3]]
            allTop5Preds.append(tempList)    
           
        elif Top5Conf[3] < new_individual_thres:
           
            tempList=[predictTop[0],predictTop[1],predictTop[2],'new_individual',predictTop[3]]        
            allTop5Preds.append(tempList)  
           
        elif Top5Conf[4] < new_individual_thres:

            tempList=[predictTop[0],predictTop[1],predictTop[2],predictTop[3],'new_individual']        
            allTop5Preds.append(tempList)        
           
        else:
            allTop5Preds.append(predictTop)

        if (('new_individual' in allTop5Preds[-1]) and (valid_labels_list[i] not in train_labels)):
            allTop5Preds[-1] = [valid_labels_list[i] if x=='new_individual' else x for x in allTop5Preds[-1]]

    metric = count_metric(valid_labels_list, allTop5Preds)
    return metric



def findLR(model, optimizer, criterion, trainloader, final_value=10, init_value=1e-8):
    '''
      findLR plots the graph for the optimum learning rates for the model with the 
      corresponding dataset.
      The technique is quite simple. For one epoch,
      1. Start with a very small learning rate (around 1e-8) and increase the learning rate linearly.
      2. Plot the loss at each step of LR.
      3. Stop the learning rate finder when loss stops going down and starts increasing.
      
      A graph is created with the x axis having learning rates and the y axis
      having the losses.
      
      Arguments:
      1. model -  (torch.nn.Module) The deep learning pytorch network.
      2. optimizer: (torch.optim) The optimiser for the model eg: SGD,CrossEntropy etc
      3. criterion: (torch.nn) The loss function that is used for the model.
      4. trainloader: (torch.utils.data.DataLoader) The data loader that loads data in batches for input into model 
      5. final_value: (float) Final value of learning rate
      6. init_value: (float) Starting learning rate.
      
      Returns:
       Nothing
       
      Plots a matplotlib graph
      
    '''
    model.train() # setup model for training configuration
    
    num = len(trainloader) - 1 # total number of batches
    mult = (final_value / init_value) ** (1/num)
    
    losses = []
    lrs = []
    best_loss = 0.
    avg_loss = 0.
    beta = 0.98 # the value for smooth losses
    lr = init_value
    
    for batch_num, (inputs, targets) in enumerate(tqdm(trainloader)):
        if len(inputs.shape) == 5:
            inputs = inputs.squeeze(1)
        
        optimizer.param_groups[0]['lr'] = lr
        
        batch_num += 1 # for non zero value
        inputs, targets = inputs.to('cuda'), targets.to('cuda') # convert to cuda for GPU usage
        optimizer.zero_grad() # clear gradients
        outputs = model(inputs) # forward pass
        loss = criterion(outputs, targets) # compute loss
       
        #Compute the smoothed loss to create a clean graph
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        
        # append loss and learning rates for plotting
        lrs.append(math.log10(lr))
        losses.append(smoothed_loss)
        
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        
        # backprop for next step
        loss.backward()
        optimizer.step()
        
        # update learning rate
        lr = mult*lr
        
    plt.xlabel('Learning Rates')
    plt.ylabel('Losses')
    plt.plot(lrs,losses)
    plt.show()


def run(config, fold):
    """Main function."""

    if config.logging.log:
        wandb.init(project=config.logging.wandb_project_name)

    if not os.path.exists(config.paths.save_dir):
        os.makedirs(config.paths.save_dir, exist_ok=True)

    with open(os.path.join(config.paths.save_dir, "log.txt"), 'w') as _: pass

    torch.cuda.empty_cache()
    model = LesGoNet(config.model_params)
    model.to(config.training.device)

    scaler = amp.GradScaler()

    criterion = ArcFaceLoss(**config.criterion_params)
    # criterion = ArcFaceLossAdaptiveMargin(margins=get_fold_margins(config, fold))

    optimizer = MADGRAD(model.parameters(), lr=5e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.training.num_epochs, eta_min=1e-7)
    # scheduler = GradualWarmupSchedulerV2(optimizer=optimizer, multiplier=100, total_epoch=1, after_scheduler=scheduler)

    best_metric = 0
    early_stopping = 0
    start_epoch = 0

    if config.paths.path_to_checkpoint is not None:
        print("Loading model from checkpoint")
        cp = torch.load(config.paths.path_to_checkpoint)

        scaler.load_state_dict(cp["scaler"])
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        for _ in range(cp["epoch"]):
            scheduler.step()

        early_stopping = cp["epochs_since_improvement"]
        start_epoch = cp["epoch"]

        del cp

    elif config.paths.path_to_pretrain is not None:
        print("Loading model from pretrain")
        model.load_state_dict(torch.load(config.paths.path_to_pretrain), strict=False)

    print("Have a nice training!")
    for epoch in range(start_epoch, config.training.num_epochs):
        print("\nEpoch:", epoch + 1)
        start_time = time.time()

        train_loader, _ = get_loaders(config, epoch, fold, is_train=True)
        # findLR(model, optimizer, criterion, train_loader, final_value=1e-2, init_value=1e-8)
        train_loss = train_loop(config, model, train_loader, optimizer, criterion, scaler)

        train_loader, val_loader = get_loaders(config, epoch, fold, is_train=False)
        metric = val_loop(config, model, val_loader, train_loader)

        scheduler.step()

        t = int(time.time() - start_time)

        if metric > best_metric:
            print("New record!")
            best_metric = metric
            early_stopping = 0
            save_model(config, model, epoch + 1, train_loss, metric, optimizer,
                         early_stopping, scheduler, scaler, True)
        else:
            early_stopping += 1
            save_model(config, model, epoch + 1, train_loss, metric, optimizer,
                         early_stopping, scheduler, scaler, False)

        print_report(t, train_loss, metric, best_metric, optimizer.param_groups[0]['lr'])
        save_log(os.path.join(config.paths.save_dir, "log.txt"), epoch, train_loss, metric, best_metric)

        if early_stopping >= config.training.early_stopping:
            print("Training has been interrupted because of early stopping.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    return best_metric

def run_eval(config, checkpoint_path, fold):

    torch.cuda.empty_cache()
    model = LesGoNet(config.model_params)
    model.to(config.training.device)

    print("Loading model from checkpoint")
    cp = torch.load(checkpoint_path)
    model.load_state_dict(cp["model"])
    del cp

    train_loader, val_loader = get_loaders(config, epoch=0, fold=fold, is_train=False)

    metric = val_loop(config, model, val_loader, train_loader)
    print('MAP@5:', metric)
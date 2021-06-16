import torch
import torch.nn as nn
from torch.autograd import Variable,Function
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import util
import argparse
import matplotlib.pyplot as plt

class BinaryActivationClass(Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension
    '''
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input #tensor.Forward should has only one output, or there will be another grad
    
    @classmethod
    def Mean(cls,input):
        return torch.mean(input.abs(),1,keepdim=True) #the shape of mnist data is (N,C,W,H)

    @staticmethod
    def backward(ctx,grad_output): #grad_output is a Variable
        input,=ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input #Variable

BinActive = BinaryActivationClass.apply

def ParseArgs():
    parser = argparse.ArgumentParser(description='XnorNet Pytorch MNIST Example.')
    parser.add_argument('--batch-size',type=int,default=100,metavar='N',
                        help='batch size for training(default: 100)')
    parser.add_argument('--test-batch-size',type=int,default=100,metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs',type=int,default=13,metavar='N',
                        help='number of epoch to train(default: 100)')
    parser.add_argument('--lr-epochs',type=int,default=20,metavar='N',
                        help='number of epochs to decay learning rate(default: 20)')
    parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum',type=float,default=0.9,metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-5,metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed',type=int,default=1,metavar='S',
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval',type=int,default=100,metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

class BinConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=False):
        super(BinConv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.layer_type = 'BinConv2d'

        self.bn = nn.BatchNorm2d(in_channels,eps=1e-4,momentum=0.1,affine=True)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,
                            groups=groups,bias=bias)
        self.relu = nn.ReLU()

    def forward(self,x):
        #block structure is BatchNorm -> BinaryActivationClass -> BinConv -> Relu
        x = self.bn(x)
        A = BinaryActivationClass().Mean(x)
        x = BinActive(x)
        k = torch.ones(1,1,self.kernel_size,self.kernel_size).mul(1/(self.kernel_size**2)) #out_channels and in_channels are both 1.constrain kernel as square
        k = Variable(k.cuda())
        K = F.conv2d(A,k,bias=None,stride=self.stride,padding=self.padding,dilation=self.dilation)
        x = self.conv(x)
        x = torch.mul(x,K)
        x = self.relu(x)
        return x

class BinLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(BinLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = nn.BatchNorm1d(in_features,eps=1e-4,momentum=0.1,affine=True)
        self.linear = nn.Linear(in_features,out_features,bias=False)

    def forward(self,x):
        x = self.bn(x)
        beta = BinaryActivationClass().Mean(x).expand_as(x)
        x = BinActive(x)
        x = torch.mul(x,beta)
        x = self.linear(x)
        return x

class LeNet5_Bin(nn.Module):
    def __init__(self):
        super(LeNet5_Bin,self).__init__()
        self.conv1 = BinConv2d(1,6,kernel_size = 5)
        self.conv2 = BinConv2d(6,16,kernel_size = 3)
        self.fc1 = BinLinear(400,50)
        self.fc2 = BinLinear(50,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = x.view(-1,400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    args = ParseArgs()
    if args.cuda:
        print("GPU AVAILABLE: ", args.cuda)
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    learning_rate = args.lr
    #momentum = args.momentum
    weight_decay = args.weight_decay

    ###################################################################
    ##             Load Train Dataset                                ##
    ###################################################################
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True,**kwargs)
    ###################################################################
    ##             Load Test Dataset                                ##
    ###################################################################
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True,**kwargs)
    model = LeNet5_Bin()
    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    #optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    bin_op = util.Binop(model)

    best_acc = 0.0 
    loss = []
    for epoch_index in range(1,args.epochs+1):
        adjust_learning_rate(learning_rate,optimizer,epoch_index,args.lr_epochs)
        loss.append(train(args,epoch_index,train_loader,model,optimizer,criterion,bin_op))
        acc = test(model,test_loader,bin_op,criterion)
        if acc > best_acc:
            best_acc = acc
            bin_op.Binarization()
            save_model(model,best_acc)
            bin_op.Restore()
            
    epochs = [i for i in range(1,args.epochs+1)]
    plt.plot(epochs, loss, 'b', label='Training accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
 
def save_model(model,acc):
    
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state,'model_state.pkl')
    

def train(args,epoch_index,train_loader,model,optimizer,criterion,bin_op):
    model.train()
    print('Training in Epoch: {}'.format(epoch_index))
    for batch_idx,(data,target) in enumerate(train_loader):
        if args.cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)

        optimizer.zero_grad()

        bin_op.Binarization()

        output = model(data)
        loss = criterion(output,target)
        loss.backward()

        bin_op.Restore()
        bin_op.UpdateBinaryGradWeight()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Processing: {:.0f}%'.format(100. * batch_idx / len(train_loader)))
            # print('Training in Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch_index, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.data))
            
    print('Loss in Epoch {} : {}'.format(epoch_index, loss.data))
    return loss.data.cpu().numpy()

def test(model,test_loader,bin_op,criterion):
    model.eval()
    test_loss = 0
    correct = 0

    bin_op.Binarization()
    for data,target in test_loader:
        data,target = data.cuda(),target.cuda()
        data,target = Variable(data,volatile=True),Variable(target)
        output = model(data)
        test_loss += criterion(output,target).data
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    bin_op.Restore()
    
    acc = 100. * correct/len(test_loader.dataset)

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return acc
    
def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr

if __name__ == '__main__':
    main()

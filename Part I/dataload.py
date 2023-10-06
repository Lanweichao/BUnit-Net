import torch
import torchvision
import torchvision.transforms as transforms

def data_load(args):
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	if args.dataname.lower() == 'cifar10':
		print('USING CIFAR10.....')
		train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
		test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
		class_num = 10
	elif args.dataname.lower() == 'cifar100':
		print('USING CIFAR100.....')
		train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
		test_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
		class_num = 100
	elif args.dataname.lower() == 'mnist':
		print('USING MNIST.....')
		train_set = torchvision.datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
		test_set = torchvision.datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)
		class_num = 10
	else:
		print('NOT IMPLEMENT! RETYPE CORRECT DATASET: {CIFAR10, CIFAR100, MNIST}')

	trainloader = torch.utils.data.DataLoader(train_set,
    										  batch_size=args.batch_size, 
    										  shuffle=True, 
    										  pin_memory=True, 
    										  num_workers=4)
	testloader = torch.utils.data.DataLoader(test_set,
    										 batch_size=args.batch_size*2, 
    										 shuffle=False, 
    										 pin_memory=True, 
    										 num_workers=4)
	return trainloader, testloader, class_num
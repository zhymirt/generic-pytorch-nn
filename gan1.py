from neural import *
from classifier import classifier
import random
use_cuda = False
device = torch.device('cuda') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
print("Cuda available? ", torch.cuda.is_available())
random.seed(1)

# def discrim_loss_fn(input_tuple, output_tuple):

class GAN(neural):

    def __init__(self, discrim=None, gen=None):
        super(GAN, self).__init__()
        assert(discrim is None or isinstance(discrim, neural))
        assert(gen is None or isinstance(gen, neural))
        self.discrim = discrim
        self.gen = gen

    def forward(self, x, single_entry=False):
        return self.gen.forward(x, single_entry)

    def fit(self, train_loader, valid_loader=None, discrim_trained=False, epochs=1, k=1):
        for i in range(epochs):
            # print(i)
            # train discrim
            # print('train discriminator')
            size = train_loader.batch_size
            for _, (discrim_x, discrim_y) in zip(range(k), train_loader):
                gen_x = self.generate_values(size)
                gen_y = torch.tensor([[1.0, 0.0] for _ in range(size)])
                self.train_discrim(gen_x, gen_y, discrim_x, discrim_y)
            # train generator
            # print('train generator')
            for _, (discrim_x, discrim_y) in zip(range(k), train_loader):
                gen_x = self.generate_values(size)
                gen_y = torch.tensor([[1.0] for _ in range(size)])
                self.train_gen(gen_x, gen_y)

    def train_discrim(self, gen_in, gen_correct, discrim_in, discrim_correct):
        self.discrim.train()
        self.discrim.optimizer.zero_grad()
        self.discrim.compute_loss(discrim_in, discrim_correct, is_guess=False, training=False)
        self.discrim.loss.backward()
        self.discrim.compute_loss(gen_in, gen_correct, is_guess=False, training=False)
        self.discrim.loss.backward()
        self.discrim.optimizer.step()

    def train_gen(self, gen_in, gen_correct):
        self.gen.train()
        gen_scores = torch.tensor([ [val[0]] for val in self.discrim.forward(gen_in, single_entry=False)])
        self.gen.compute_loss(gen_scores, gen_correct, training=True)

    def generate_values(self, num_values):
        random_values = torch.tensor([random.gauss(0, 10)*100 for _ in range(num_values)], dtype=torch.float)
        return torch.tensor([ [self.gen.forward(torch.tensor([x]), single_entry=(num_values == 1))] for x in random_values])

    def compute_loss(self, input_data, correct, is_guess=False, training=False):
        if is_guess:
            return self.gen.compute_loss(input_data, correct, is_guess=True, training=training)
        else:
            input_scores = torch.tensor([ [val[0]] for val in self.discrim.forward(input_data, single_entry=False)])
            return self.gen.compute_loss(input_scores, correct, is_guess=True, training=training)

    def save(self, filename):
        torch.save()


gen_layers = OrderedDict([
    ('fc1', nn.Linear(1, 8)),
    ('r1', nn.ReLU()),
    ('fc2', nn.Linear(8, 16)),
    ('r2', nn.ReLU()),
    ('fc3', nn.Linear(16, 64)),
    ('r3', nn.ReLU()),
    ('fc4', nn.Linear(64, 16)),
    ('r4', nn.ReLU()),
    ('fc5', nn.Linear(16, 1))
])
discrim_layers = OrderedDict([
    ('fc1', nn.Linear(1, 8)),
    ('r1', nn.ReLU()),
    ('fc2', nn.Linear(8, 16)),
    ('r2', nn.ReLU()),
    ('fc3', nn.Linear(16, 64)),
    ('r3', nn.ReLU()),
    ('fc4', nn.Linear(64, 16)),
    ('r4', nn.ReLU()),
    ('fc5', nn.Linear(16, 1))   
])
gen_layers2 = deepcopy(gen_layers)
discrim_layers2 = OrderedDict([
    ('fc1', nn.Linear(1, 8)),
    ('r1', nn.ReLU()),
    ('fc2', nn.Linear(8, 16)),
    ('r2', nn.ReLU()),
    ('fc3', nn.Linear(16, 64)),
    ('r3', nn.ReLU()),
    ('fc4', nn.Linear(64, 16)),
    ('r4', nn.ReLU()),
    ('fc5', nn.Linear(16, 2))   
])
classes = ('fake', 'real')
data_in = torch.randint(low=0, high=50, size=[10000,1], dtype=torch.float32, device=device)
data_in = torch.mul(data_in, 2)
data_in = torch.add(data_in, 1)
# data_in = data_in.pow(3)
print(data_in)
# data_out = torch.ones([10000, 1])
data_out = torch.tensor([[0.0, 1.0] for _ in range(10000)])
print(data_out)
data_dataset = TensorDataset(data_in, data_out)
train_data, validate_data, test_data = torch.utils.data.random_split(data_dataset, [8000, 1000, 1000])
train_data_loader = DataLoader(train_data, batch_size=8000, shuffle=True)
validate_data_loader = DataLoader(validate_data, batch_size=4)
test_data_loader = DataLoader(test_data, batch_size=1000)

gen = neural(layers=gen_layers2)
discrim = classifier(layers=discrim_layers2)
discrim.set_loss_function(nn.BCEWithLogitsLoss())
gan1 = GAN(discrim=discrim, gen=gen)

print(gan1.generate_values(10))
gan1.fit(train_data_loader, validate_data_loader, epochs=1000, k=1)
print(gan1.generate_values(1))
print(gan1.compute_loss(gan1.generate_values(1), torch.tensor([[1.0]]), is_guess=False))
# print(gan1.generate_values(1))
print(gan1.generate_values(10))
# fake_vals = torch.tensor([[random.randint(1, 100)] for _ in range(10)])
# print('fake_values: ', gan1.gen.forward(fake_vals))
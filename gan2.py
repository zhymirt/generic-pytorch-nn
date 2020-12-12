""" GAN attempt 2"""
import torch
from neural import Neural
from classifier import Classifier
from collections import OrderedDict


def generator_loss(output, target):
    loss = torch.log
class Generator(Neural):
    def train_with_loader(self, data, validating_data=None, scheduler=None, epochs=1):
        """
        Trains network using dataloaders with optional validation dataloader.
        """
        print('Training...')
        for epoch in range(epochs):
            self.train()
            for train_in, train_out in data:
                self.compute_loss(train_in, train_out, is_guess=False, training=True)
            self.eval()
            if validating_data:
                with torch.no_grad():
                    valid_loss = self.compute_loss_loader(validating_data).item()
                    print('Average validation error at step ',epoch+1,': ', valid_loss)
                if scheduler and valid_loss:
                    scheduler.step()

    def compute_loss_loader(self, test_data):
        with torch.no_grad():
            total_loss = 0
            self.eval()
            for test_in, test_out in test_data:
                total_loss += self.compute_loss(test_in, test_out, is_guess=False, training=False)
            return total_loss/len(test_data) if test_data else None

    def compute_loss(self, input_data, correct, is_guess=False, training=False):
        input_data = input_data.to(self.device) if is_cuda(self.device) else input_data
        correct = correct.to(self.device) if is_cuda(self.device) else correct
        if is_guess:
            self.loss = self.loss_fn(input_data,correct)
        else:
            self.loss = self.loss_fn(self.forward(input_data),correct)
        if training:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        return self.loss


class GAN(Neural):

    def __init__(self, discrim=None, gen=None):
        super(GAN, self).__init__()
        assert(discrim is None or isinstance(discrim, Neural))
        assert(gen is None or isinstance(gen, Neural))
        self.discrim = discrim
        self.gen = gen

    def forward(self, x, single_entry=False):
        return self.gen.forward(x, single_entry)

    def fit(self, generator_noise, real_data, epochs):
        # have generator make fake data
        # discriminator learns to tell fake from real
        # generator learns to make better fakes
        # repeat loop until discriminator is 50/50 guessing
        for i in range(epochs):
            gen_data = self.gen.forward(generator_noise)
            self.discrim.train_with_loader




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


def train_gan():
    generator = Neural()
    discrim = Classifier()
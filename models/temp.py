
# if self.conv2d is None:
#     # in_channels = x.size(1)
#     # self.conv2d1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, padding=0)
#     self.conv2d2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=self.stride,padding=0)
#     # self.conv2d1 = self.conv2d1.to(x.device)
#     self.conv2d2 = self.conv2d2.to(x.device)
# cur_former = self.cur
# self.neuronal_charge(x)
# spike = self.neuronal_fire()  # LIF fire
# self.neuronal_reset(spike)  # LIF reset
# self.vot += spike * torch.sigmoid(self.v)
# self.v = self.v - spike * torch.sigmoid(self.vot)  # Reset
# self.cur = torch.tanh(self.conv2d2(spike))
# return spike, cur_former if isinstance(cur_former, torch.Tensor) else torch.tanh(torch.zeros_like(self.cur))
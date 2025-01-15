class PCNNs:
    def __init__(self, u):
        self.u = u
        self.lambda_1 = torch.tensor([0.1], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_3 = torch.tensor([0.1], requires_grad=True).to(device)
        self.lambda_4 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_5 = torch.tensor([0.3], requires_grad=True).to(device)
        self.lambda_6 = torch.tensor([0.1], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.lambda_3 = torch.nn.Parameter(self.lambda_2)
        self.lambda_4 = torch.nn.Parameter(self.lambda_2)
        self.lambda_5 = torch.nn.Parameter(self.lambda_2)
        self.lambda_6 = torch.nn.Parameter(self.lambda_2)

        self.net = DNN(dim_in=2, dim_out=1, n_layer=7, n_node=20, ub=ub, lb=lb,).to(device)
        self.net.register_parameter("lambda_1", self.lambda_1)
        self.net.register_parameter("lambda_2", self.lambda_2)
        self.net.register_parameter("lambda_2", self.lambda_3)
        self.net.register_parameter("lambda_2", self.lambda_4)
        self.net.register_parameter("lambda_2", self.lambda_5)
        self.net.register_parameter("lambda_2", self.lambda_6)
        # self.net.register_parameter("lambda_2", self.lambda_2)
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0

    def f(self, xt):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        lambda_4 = self.lambda_4
        lambda_5 = self.lambda_5
        lambda_6 = self.lambda_6
        # lambda_1 = self.lambda_1

        xt = xt.clone()
        xt.requires_grad = True

        u = self.net(xt)
        #
        # u_xt = grad(u.sum(), xt, create_graph=True)[0]
        # u_x = u_xt[:, 0:1]
        # u_t = u_xt[:, 1:2]
        #
        # u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        # f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        f = lambda_1 * dayl  + lambda_2 * prcp + lambda_3 * srad +lambda_4 *tmax+lambda_5 *tmin +lambda_6 *vp + x
        # Apply output bounds
        f = torch.clamp(f, 0, 365)
        return f


    def closure(self):
        self.optimizer.zero_grad()
        u_pred = self.net(xt)
        f_pred = self.f(xt)
        mse_u = torch.mean(torch.square(u_pred - self.u))
        mse_f = torch.mean(torch.square(f_pred))
        loss = mse_u + mse_f
        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e} l1 : {self.lambda_1.item():.5f}, l2 : {torch.exp(self.lambda_2).item():.5f}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss

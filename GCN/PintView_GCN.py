from Model import Model
import torch
import torch.nn as nn
from utils import View_selector, LocalGCN, NonLocalMP

class PointViewGCN(Model):
    def __init__(self, name, nclasses=40, num_views=20):
        super(PointViewGCN, self).__init__(name)
        self.nclasses = nclasses
        self.num_views = num_views
        self.drop1 = nn.Dropout(0.5)

        vertices = [[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], [1.44337567, -1.44337567, -1.44337567],
                    [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567],
                    [0, 0.89205522, 2.3354309], [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309],
                    [2.3354309, 0, 0.89205522], [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522],
                    [0.89205522, 2.3354309, 0], [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]]

        self.num_views_mine = 60
        self.vertices = torch.tensor(vertices).cuda()
        self.LocalGCN1 = LocalGCN(k=4, n_views=self.num_views_mine // 3)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views_mine // 3)
        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views_mine // 4)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views_mine // 4)
        self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views_mine // 6)
        self.NonLocalMP3 = NonLocalMP(n_view=self.num_views_mine // 6)
        self.LocalGCN4 = LocalGCN(k=4, n_views=self.num_views_mine // 12)

        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views_mine // 4)
        self.View_selector2 = View_selector(n_views=self.num_views_mine // 4, sampled_view=self.num_views_mine // 6)
        self.View_selector3 = View_selector(n_views=self.num_views_mine // 6, sampled_view=self.num_views_mine // 12)

        self.cls = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.nclasses)
        )
        # print(self.cls)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        views = self.num_views
        y = x
        y = y.view((int(x.shape[0] / views), views, -1))
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)

        y = self.LocalGCN1(y, vertices)
        # sahep y [1,20,512]
        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]

        z, F_score, vertices2 = self.View_selector1(y2, vertices, k=4)
        # shape z = [1,10,512]
        z = self.LocalGCN2(z, vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]
        # shape pooled_view2 [1,512]

        m, F_score_m, vertices_m = self.View_selector2(z2, vertices2, k=4)
        m = self.LocalGCN3(m, vertices_m)
        m2 = self.NonLocalMP3(m)
        pooled_view3 = torch.max(m, 1)[0]
        # pooled_view3 = pooled_view1 + pooled_view3

        w, F_score2, vertices3 = self.View_selector3(m2, vertices_m, k=4)
        w = self.LocalGCN4(w, vertices3)
        pooled_view4 = torch.max(w, 1)[0]
        # pooled_view4 = pooled_view4 + pooled_view1

        pooled_view = torch.cat((pooled_view1, pooled_view2, pooled_view3, pooled_view4), 1)
        pooled_view = self.cls(pooled_view)

        return pooled_view, F_score, F_score_m, F_score2
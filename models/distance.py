from Code.models.Wasserstein import SinkhornDistance

def distance_(p, q ,args):
    args = args
    if args.dis_type == "KL":
        # # kl --------Data_set
        return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)
    elif args.dis_type == "BD":
        # # Bhattacharyya Distance --------2
        return  (-1)*((((p * q)**0.5).sum()).log())
    elif args.dis_type == "JS":
        # # Jensen-Shannon divergence --------3
        # M = (p + q) / 2
        # # return 0.5*(p*(p / M).log()).sum() + 0.5*(q*((q / M).log())).sum()
        return 1/2.0*((p*((p + 1e-5).log()-((p+q+2*(1e-5))/2.0).log())).sum()) + 1/2.0*((q*((q + 1e-5).log()-((p+q+2*(1e-5))/2.0).log())).sum())
    elif args.dis_type == "WD":
        # #wasserstein distance  --------6
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=30)
        dist, P, C = sinkhorn(p, q)
        return dist
    elif args.dis_type == "HD":
        # # Hellinger distance --------7
        return ((((p ** 0.5 - q ** 0.5) ** 2).sum()) ** 0.5) * (1.0 / (2 ** 0.5))
    elif args.dis_type == "ED":
        # #Euclidean Distance --------4
        return (((p - q) ** 2).sum()) ** 0.5
    elif args.dis_type == "CE":
        # # crocess entropy --------5
        return (p * (q.log())).sum(-1)
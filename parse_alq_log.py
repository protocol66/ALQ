import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='alq_log.txt')
    parser.add_argument('--output_file', type=str, default='alq_log.csv')
    args = parser.parse_args()
    
    with open(args.log_file, 'r') as f:
        with open(args.output_file, 'w') as f_out:
            lines = f.readlines()
            for l in lines:
                # which step
                if re.search(r'initialization', l):
                    # init = True
                    # basis = False
                    # coord = False
                    # prune = False
                    step = 'init'
                elif re.search(r'basis', l):
                    # basis = True
                    # init = False
                    # coord = False
                    # prune = False
                    step = 'basis'
                elif re.search(r'coordinate', l):
                    # coord = True
                    # init = False
                    # basis = False
                    # prune = False
                    step = 'coord'
                elif re.search(r'pruning', l):
                    # prune = True
                    # init = False
                    # basis = False
                    # prune = False
                    step = 'prune'
                
                # if init:
                #     if re.match(r'train loss:', l):
                #         loss = re.search(r'[0-9]+\.[0-9]+', l)[0]
                #         f_out.write(f'init, {loss}')
                
                if re.search(r'((train)|(prun))(ing)? loss', l):
                    loss = re.search(r'[0-9]+\.[0-9]+', l)[0]
                    f_out.write(f'{step}, {loss}\n')
                
                    
                    
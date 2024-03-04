import argparse

parser = argparse.ArgumentParser(
                    prog = 'Cluster example',
                    description = 'Prints Hello World.',
                    epilog = 'Have fun :).')

parser.add_argument('--job_id')

args = parser.parse_args()
job_id = args.job_id
print("Hello World from job {}!".format(job_id))

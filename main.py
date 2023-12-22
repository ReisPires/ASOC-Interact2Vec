import sys

from recresearch.experiments import ExperimentalProtocol

def main():
    experimental_protocol = ExperimentalProtocol(sys.argv)
    experimental_protocol.run()

if __name__ == '__main__':
    main()
    sys.exit(0)
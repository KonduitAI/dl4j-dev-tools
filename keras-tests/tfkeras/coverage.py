from utils import get_coverage

coverage = get_coverage()

print('Layers not covered: ')
for l in coverage['uncovered_layers']:
    print(l)

print('Layer coverage: ' +
      str(int(100 * len(coverage['covered_layers']) / len(coverage['all_layers']))) + '% (' +
      str(len(coverage['covered_layers'])) + '/' + str(len(coverage['all_layers'])) + ')')

from collections import defaultdict
from oswegonlp import constants

theta_manual = {('fake','media'):0.2,
                ('fake','hillary'):0.1,
                ('fake','obama'):0.2,
                ('real','signs'):0.3,
                ('real','tweets'):0.1,
                ('real','travel'):0.2,
                ('real',constants.OFFSET):0.01}

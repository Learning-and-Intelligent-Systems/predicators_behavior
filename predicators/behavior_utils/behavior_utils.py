"""Utility functions for using iGibson and BEHAVIOR."""

import logging
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
import pybullet as p
from tqdm import tqdm

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Array, GroundAtom, GroundAtomTrajectory, \
    LowLevelTrajectory, Predicate, Set, State, Task

try:
    from igibson import object_states
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.object_states.on_floor import \
        RoomFloor  # pylint: disable=unused-import
    from igibson.objects.articulated_object import URDFObject
    from igibson.robots.behavior_robot import \
        BRBody, BehaviorRobot  # pylint: disable=unused-import
    from igibson.robots.robot_base import \
        BaseRobot  # pylint: disable=unused-import
    from igibson.utils.checkpoint_utils import load_checkpoint
except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
    pass

# NOTE: Each of these below constants obtained from parsing script in
# LIS fork of the bddl repository. These sets might be incomplete, so
# if you encounter an error while trying to solve a new BEHAVIOR task,
# you might need to add to these.
ALL_RELEVANT_OBJECT_TYPES = {
    'breakfast_table', 'notebook', 'hardback', 'shelf', 'agent', 'room_floor',
    'coffee_table', 'pop', 'bed', 'bucket', 'plate', 'hamburger', 'countertop',
    'trash_can', 'backpack', 'toothbrush', 'shampoo', 'underwear', 'door',
    'window', 'toothpaste', 'package', 'highlighter', 'swivel_chair',
    'document', 'bottom_cabinet_no_top', 'folder', 'bottom_cabinet',
    'top_cabinet', 'sofa', 'oatmeal', 'chip', 'vegetable_oil', 'sugar',
    'cabinet', 'floor', 'pasta', 'sauce', 'electric_refrigerator', 'olive_oil',
    'sugar_jar', 'spaghetti_sauce', 'mayonnaise', 'fridge', 'board_game',
    'video_game', 'facsimile', 'ice_cube', 'ginger', 'gym_shoe', 'car', 'mug',
    'toy', 'hanger', 'candle', 'basket', 'spoon', 'tile', 'pool', 'sock',
    'straight_chair', 'sticky_note', 'carton', 'cologne', 'cup', 'ball',
    'lipstick', 'newspaper', 'pretzel', 'necklace', 'candy_cane', 'briefcase',
    'sushi', 'sweater', 'rag', 'scrub_brush', 'cookie', 'bowl', 't-shirt',
    'cheese', 'detergent', 'catsup', 'pencil_box', 'bracelet', 'saucepan',
    'soap', 'wine_bottle', 'dishwasher', 'lollipop', 'cinnamon', 'pen', 'sink',
    'bow', 'bath_towel', 'cruet', 'headset', 'coffee_cup', 'dishtowel',
    'mouse', 'stove', 'duffel_bag', 'broom', 'stocking', 'parsley', 'yogurt',
    'guacamole', 'paper_towel', 'modem', 'scanner', 'printer', 'mousetrap',
    'toilet'
}
PICK_PLACE_OBJECT_TYPES = {
    'mineral_water',
    'oatmeal',
    'blueberry',
    'headset',
    'jug',
    'flank',
    'baseball',
    'crab',
    'dressing',
    'cranberry',
    'trout',
    'kale',
    'shoe',
    'licorice',
    'decaffeinated_coffee',
    'cookie',
    'whiskey',
    'bench',
    'alcohol',
    'journal',
    'fork',
    'cherry',
    'tin',
    'winter_melon',
    'cocktail',
    'pretzel',
    'bidet',
    'bowl',
    'nectarine',
    'dish',
    'baked_goods',
    'noodle',
    'gingerbread',
    'lemonade',
    'basket',
    'sock',
    'brandy',
    'apricot',
    'plum',
    'diary',
    'cabbage',
    'cupcake',
    'lentil',
    'turnip',
    'solanaceous_vegetable',
    'straight_chair',
    'pomegranate',
    'pastry',
    'sugar',
    'marble',
    'poultry',
    'backpack',
    'tank',
    'helmet',
    'cornbread',
    'ribbon',
    'pitcher',
    'caramel',
    'folding_chair',
    'hamper',
    'sushi',
    'hot_sauce',
    'cheeseboard',
    'scoop',
    'highchair',
    'mascara',
    'newspaper',
    'punch',
    'liqueur',
    'beef',
    'jersey',
    'julienne',
    'marker',
    'root_vegetable',
    'cantaloup',
    'screwdriver',
    'softball',
    'porterhouse',
    'lipstick',
    'highlighter',
    'soap',
    'candle',
    'ring',
    'ale',
    'banana',
    'toothpaste',
    'coffeepot',
    'toothbrush',
    'carving_knife',
    'meat_loaf',
    'plaything',
    'cake',
    'bagel',
    'freshwater_fish',
    'lobster',
    'asparagus',
    'quick_bread',
    'olive',
    'date',
    'sweatshirt',
    'sprout',
    'candy',
    'corn_chip',
    'coconut',
    'blush_wine',
    'carrot',
    'grapefruit',
    'mozzarella',
    'prawn',
    'parmesan',
    'dentifrice',
    'papaya',
    'frying_pan',
    'hardback',
    'head_cabbage',
    'potholder',
    'currant',
    'mostaccioli',
    'coffee',
    'bracelet',
    'flatbread',
    'broccoli',
    'dipper',
    'mocha',
    'tea',
    'loaf_of_bread',
    'rump',
    'plate',
    'vidalia_onion',
    'eyeshadow',
    'mandarin',
    'painting',
    'footstool',
    'chip',
    'apple',
    'tenderloin',
    'sack',
    'orange_liqueur',
    'cucumber',
    'snowball',
    'lime',
    'rocking_chair',
    'french_bread',
    'stool',
    'teapot',
    'loafer',
    'gumbo',
    'salami',
    'granola',
    'perfume',
    'frisbee',
    'alarm',
    'chickpea',
    'bun',
    'flour',
    'stiletto',
    'oxford',
    'muskmelon',
    'macaroni',
    'armor_plate',
    'wine_bottle',
    'mouse',
    'lettuce',
    'gouda',
    'sausage',
    'fennel',
    'salmon',
    'crock',
    'eggplant',
    'potpourri',
    'cardigan',
    'saltwater_fish',
    'pad',
    'cola',
    'bead',
    'venison',
    'fruit_drink',
    'blanc',
    'coca_cola',
    'brie',
    'butter',
    'jewelry',
    'pencil_box',
    'anchovy',
    'grape',
    'cold_cereal',
    'dagger',
    'cologne',
    'chair',
    'gazpacho',
    'wine',
    'shellfish',
    'biscuit',
    'red_salmon',
    'planner',
    'bath',
    'espresso',
    'pepperoni',
    'brush',
    'muffin',
    'clamshell',
    'baguet',
    'penne',
    'radish',
    'cracker',
    'chop',
    'pen',
    'drum',
    'mat',
    'bathtub',
    'edible_fruit',
    'curacao',
    'scotch',
    'sparkling_wine',
    'pie',
    'pineapple',
    'tomato',
    'sheet',
    'gumdrop',
    'sharpie',
    'pullover',
    'smoothie',
    'snack_food',
    'cut_of_beef',
    'candy_cane',
    'tablespoon',
    'sofa',
    'raspberry',
    'pencil',
    'coleslaw',
    'pancake',
    'farfalle',
    'tortilla',
    'chili',
    'spinach',
    'umbrella',
    'peach',
    'groundsheet',
    'vegetable_oil',
    'silver_salmon',
    'mousetrap',
    'feijoa',
    'clam',
    'knife',
    'tray',
    'green_onion',
    'armchair',
    'crouton',
    'duffel_bag',
    'cos',
    'necklace',
    'mattress',
    'scanner',
    'steak',
    'eyeliner',
    'greens',
    'zucchini',
    'kiwi',
    'game',
    'tablefork',
    'squash',
    'hamburger',
    'marinara',
    'lollipop',
    'ball',
    'juice',
    'worcester_sauce',
    'chaise_longue',
    'demitasse',
    'cider',
    'document',
    'cauliflower',
    'fish',
    'beverage',
    'vessel',
    'earplug',
    'percolator',
    'french_dressing',
    'sweet_pepper',
    'workwear',
    'pan',
    'mussel',
    'shank',
    'modem',
    'turkey',
    'pepper',
    'vodka',
    'rib',
    'scraper',
    'summer_squash',
    'honeydew',
    'facsimile',
    'pear',
    'ravioli',
    'dredging_bucket',
    'cheese',
    'spaghetti_sauce',
    'gorgonzola',
    'lingerie',
    'hat',
    'nightgown',
    'bird',
    'artichoke',
    'brownie',
    'pea',
    'ice_cube',
    'hanger',
    'walker',
    'doll',
    'paper_towel',
    'milk',
    'lemon',
    'mayonnaise',
    'brew',
    'parsley',
    'cellophane',
    'jar',
    'broth',
    'pop',
    'champagne',
    'wafer',
    'rum',
    'soft_drink',
    'barrel',
    'water',
    'towel',
    'caldron',
    'carafe',
    'stockpot',
    'cream_pitcher',
    'salad',
    'kettle',
    'olive_oil',
    'cut_of_pork',
    'book',
    'bow',
    'chocolate',
    'basin',
    'apparel',
    'autoclave',
    'watermelon',
    'mushroom',
    'soup',
    'bacon',
    'basketball',
    'cube',
    'telephone_receiver',
    'drinking_vessel',
    'cup',
    'fig',
    'scrub_brush',
    'bean',
    'sandal',
    'dustpan',
    'calculator',
    'white_sauce',
    'buttermilk',
    'bath_towel',
    'plastic_wrap',
    'chicory',
    'dishrag',
    'beet',
    'boiler',
    'roaster',
    'saucepan',
    'bleu',
    'cruet',
    'radicchio',
    'produce',
    'nacho',
    'tuna',
    'sirloin',
    'veal',
    'tea_bag',
    'mug',
    'ladle',
    'lamb',
    'mixed_drink',
    'sweater',
    'lego',
    'canola_oil',
    'meat',
    'orange',
    'tortilla_chip',
    'fudge',
    'shampoo',
    'cheddar',
    'breakfast_food',
    'wrapping',
    'mango',
    'nan',
    'swivel_chair',
    'catsup',
    'bread',
    'hand_towel',
    'berry',
    'makeup',
    'drinking_water',
    'shirt',
    'bell_pepper',
    'pork',
    'liquor',
    'sandwich',
    'pumpkin',
    'gym_shoe',
    'vegetable',
    'martini',
    'melon',
    'cayenne',
    'football',
    'mint',
    'album',
    'beefsteak',
    'seltzer',
    'cognac',
    'scrapbook',
    'white_bread',
    'wreath',
    'egg',
    'notebook',
    'cleansing_agent',
    'rag',
    'beer',
    'citrus',
    'folder',
    'bucket',
    'detergent',
    'chicory_escarole',
    'marshmallow',
    'earphone',
    'puppet',
    'ham',
    'dustcloth',
    'briefcase',
    'prosciutto',
    'chicken',
    'gourd',
    'tabasco',
    'mural',
    'printer',
    'tequila',
    'potato',
    'cut',
    'gravy',
    'scone',
    'cereal',
    'avocado',
    'pasta',
    'vase',
    'underwear',
    'cheesecake',
    'seafood',
    'dried_fruit',
    'shallot',
    'carton',
    'legume',
    'blackberry',
    'tabbouleh',
    'coffee_cup',
    'salad_green',
    'paintbrush',
    'brisket',
    'sunhat',
    'sunglass',
    'toast',
    'soy',
    'seat',
    'carpet_pad',
    'blender',
    'tart',
    'puff',
    'jewel',
    'strawberry',
    'onion',
    'toasting_fork',
    'bottle',
    'pomelo',
    'teacup',
    'pot',
    'food',
    'bisque',
    'hairbrush',
    'spoon',
    'novel',
    'teaspoon',
    'waffle',
    'pita',
    'yogurt',
    'stout',
    'dishtowel',
    'paperback_book',
    'casserole',
    'earring',
    'peppermint',
    'cruciferous_vegetable',
    'soup_ladle',
    'jean',
    'teddy',
    'chestnut',
    'sauce',
    'piece_of_cloth',
    'whitefish',
    'siren',
    'balloon',
    'celery',
    'hot_pepper',
    'raisin',
    'sugar_jar',
    'toy',
    'sticky_note',
    't-shirt',
    'board_game',
    'video_game',
    'ginger',
    'tile',
}
PLACE_ONTOP_SURFACE_OBJECT_TYPES = {
    'towel', 'tabletop', 'face', 'brim', 'cheddar', 'chaise_longue', 'stove',
    'gaming_table', 'rocking_chair', 'swivel_chair', 'car', 'dartboard',
    'hand_towel', 'edge', 'sheet', 'countertop', 'carton', 'deep-freeze',
    'shelf', 'floorboard', 'bookshelf', 'flatbed', 'worktable',
    'pedestal_table', 'highchair', 'side', 'sofa', 'armor_plate',
    'horizontal_surface', 'floor', 'mantel', 'table', 'gouda', 'clipboard',
    'breakfast_table', 'christmas_tree', 'dressing_table', 'basket',
    'mozzarella', 'cheese', 'truck_bed', 'screen', 'parmesan', 'gorgonzola',
    'pegboard', 'bath_towel', 'platform', 'writing_board', 'straight_chair',
    'coffee_table', 'desk', 'armchair', 'brie', 'front', 'bleu', 'helmet',
    'paper_towel', 'dishtowel', 'dial', 'folding_chair', 'deck', 'chair',
    'hamper', 'bed', 'plate', 'work_surface', 'board', 'pallet',
    'console_table', 'pool_table', 'electric_refrigerator', 'stand',
    'room_floor', 'notebook', 'hardback', 'toilet'
}
PLACE_INTO_SURFACE_OBJECT_TYPES = {
    'shelf', 'sack', 'basket', 'dredging_bucket', 'cabinet', 'crock', 'bucket',
    'casserole', 'bookshelf', 'teapot', 'dishwasher', 'deep-freeze', 'hamper',
    'bathtub', 'car', 'vase', 'jar', 'bin', 'mantel', 'stocking', 'ashcan',
    'electric_refrigerator', 'clamshell', 'backpack', 'sink', 'carton', 'dish',
    'trash_can', 'bottom_cabinet_no_top', 'fridge', 'bottom_cabinet',
    'top_cabinet'
}
OPENABLE_OBJECT_TYPES = {
    'sack', 'storage_space', 'trap', 'turnbuckle', 'lock', 'trailer_truck',
    'duplicator', 'slide_fastener', 'car', 'jug', 'percolator', 'window',
    'coupling', 'package', 'collar', 'deep-freeze', 'walnut', 'pack', 'truck',
    'nozzle', 'shoebox', 'journal', 'toolbox', 'grill', 'work', 'box', 'tin',
    'wine_bottle', 'canned_food', 'choke', 'file', 'door', 'bag', 'facsimile',
    'washer', 'envelope', 'basket', 'dredging_bucket', 'crock', 'spout',
    'carabiner', 'album', 'screen', 'diary', 'drain', 'watchband', 'armoire',
    'accessory', 'tent', 'dishwasher', 'dryer', 'vent', 'writing_board',
    'personal_computer', 'scrapbook', 'bale', 'bin', 'coil', 'egg', 'reactor',
    'belt', 'recreational_vehicle', 'canister', 'tie', 'notebook', 'backpack',
    'stopcock', 'brassiere', 'pencil_box', 'tank', 'marinade', 'material',
    'hinge', 'folder', 'wastepaper_basket', 'bucket', 'folding_chair',
    'digital_computer', 'jaw', 'drawstring_bag', 'hamper', 'caddy',
    'refrigerator', 'briefcase', 'shaker', 'hindrance', 'stapler', 'drawer',
    'jar', 'binder', 'planner', 'gear', 'cupboard', 'gourd', 'windowpane',
    'champagne', 'cooler', 'barrel', 'can', 'clamshell',
    'electric_refrigerator', 'van', 'bird_feeder', 'kit', 'plate_glass',
    'printer', 'mascara', 'velcro', 'pill', 'range_hood', 'packet', 'pen',
    'knot', 'stove', 'marker', 'wallet', 'wardrobe', 'faucet', 'polish',
    'cotter', 'bulldog_clip', 'connection', 'bundle', 'white_goods',
    'office_furniture', 'vase', 'carryall', 'lipstick', 'sparkling_wine',
    'kettle', 'highlighter', 'clamp', 'book', 'microwave', 'nutcracker',
    'banana', 'pane', 'hole', 'coffeepot', 'bow', 'carton', 'sharpie',
    'toilet', 'canopy', 'autoclave', 'pouch', 'drawstring', 'ventilation',
    'lid', 'caster', 'buckle', 'mail', 'portable_computer', 'clog', 'capsule',
    'clipboard', 'umbrella', 'packaging', 'laptop', 'drumstick', 'thing',
    'mousetrap', 'shoulder_bag', 'latch', 'movable_barrier', 'hardback',
    'chest_of_drawers', 'cage', 'novel', 'roaster', 'duffel_bag', 'diaper',
    'ashcan', 'junction', 'mechanical_system', 'crusher', 'frame', 'shelter',
    'chest', 'magazine', 'wicker', 'paperback_book', 'scanner', 'computer',
    'dose', 'clasp', 'eyeliner', 'clothespin', 'hood', 'trademark', 'pincer',
    'crate', 'cabinet', 'joint', 'bottom_cabinet_no_top', 'fridge',
    'bottom_cabinet', 'trash_can'
}

TOGGLEABLE_OBJECT_TYPES = {
    'facsimile',
}

PLACE_NEXT_TO_SURFACE_OBJECT_TYPES = {
    'toilet',
}
CLEANING_OBJECT_TYPES = {
    'toothbrush', 'towel', 'dinner_napkin', 'paper_towel', 'dishtowel',
    'broom', 'vacuum', 'rag', 'carpet_sweeper', 'hand_towel', 'scraper',
    'bath_towel', 'eraser', 'dustcloth', 'scrub_brush'
}

DUSTYABLE_OBJECT_TYPES = {
    'bottom_cabinet_no_top', 'tabletop', 'face', 'dumbbell', 'corkscrew',
    'terry', 'circle', 'fur', 'coaster', 'gauze', 'cotton', 'chock', 'trap',
    'compact_disk', 'cap', 'stake', 'converter', 'peripheral',
    'slide_fastener', 'headset', 'jug', 'baseball', 'window', 'straightener',
    'computer_game', 'toaster', 'data_input_device', 'collar', 'accelerator',
    'shoe', 'truck', 'remote_control', 'apron', 'nozzle', 'dander', 'bench',
    'washcloth', 'ipod', 'journal', 'fork', 'outerwear', 'cherry',
    'ammunition', 'work', 'quilt', 'wheel', 'tin', 'plywood', 'instrument',
    'bidet', 'bag', 'bowl', 'squeegee', 'water_scooter', 'washer', 'panel',
    'basket', 'motor_vehicle', 'mixer', 'hammer', 'suit', 'carabiner', 'sock',
    'slipper', 'screen', 'boat', 'bath_linen', 'diary', 'drain', 'armoire',
    'pegboard', 'dishwasher', 'vent', 'golf_equipment', 'wrench',
    'personal_computer', 'straight_chair', 'suede_cloth',
    'semiconductor_device', 'ginger', 'machine', 'backpack', 'tank',
    'computer_circuit', 'helmet', 'magnetic_disk', 'ribbon', 'arrangement',
    'pitcher', 'table_knife', 'disk', 'folding_chair', 'circuit', 'pump',
    'toweling', 'solid_figure', 'drawstring_bag', 'rule', 'hamper', 'caddy',
    'cheeseboard', 'console_table', 'headlight', 'medical_instrument', 'can',
    'highchair', 'musical_instrument', 'canvas', 'velcro', 'mascara',
    'loudspeaker', 'hacksaw', 'newspaper', 'packet', 'knot', 'jersey',
    'shaver', 'screw', 'marker', 'floor_lamp', 'carriage', 'act', 'tulle',
    'screwdriver', 'greatcoat', 'softball', 'cotter', 'parlor_game', 'skewer',
    'tack', 'eiderdown', 'highlighter', 'candle', 'ring', 'bell', 'banana',
    'grate', 'coffeepot', 'crib', 'flashlight', 'shrapnel', 'toothbrush',
    'inverter', 'shelf', 'regulator', 'carving_knife', 'psychological_feature',
    'camcorder', 'spring', 'razor', 'ceramic_ware', 'indentation', 'post',
    'plumbing', 'handle', 'caster', 'staple', 'buckle', 'frill', 'tumbler',
    'steamer', 'cradle', 'earmuff', 'digital_camera', 'swab', 'shield', 'hose',
    'caliper', 'activity', 'mill', 'acoustic_device', 'room', 'carrot',
    'drumstick', 'thimble', 'ladder', 'thing', 'snake', 'wedge', 'stairway',
    'voltmeter', 'heater', 'push_button', 'frying_pan', 'photograph',
    'hardback', 'basil', 'trouser', 'rail', 'folderal', 'junction', 'dress',
    'anchor', 'blade', 'change_of_location', 'chest', 'bolt', 'bracelet',
    'photographic_equipment', 'printed_circuit', 'dial', 'stringed_instrument',
    'computer', 'clasp', 'dipper', 'cornice', 'uniform', 'vacuum',
    'electro-acoustic_transducer', 'clothespin', 'hood', 'plate', 'bannister',
    'brocade', 'pincer', 'pool_table', 'eyeshadow', 'painting', 'footstool',
    'bedpost', 'power_tool', 'sifter', 'drum_sander', 'drive', 'blazer',
    'layer', 'apple', 'shaft', 'writing', 'towel_rack', 'thermostat',
    'microphone', 'rocking_chair', 'stool', 'teapot', 'turnbuckle',
    'toothpick', 'stirrer', 'loafer', 'whisk', 'sieve', 'frisbee', 'movement',
    'alarm', 'fan', 'dartboard', 'spatula', 'filter', 'gauge', 'bobbin',
    'countertop', 'electrical_converter', 'heating_element', 'sink',
    'stiletto', 'porcelain', 'product', 'roller', 'pack', 'circuit_breaker',
    'oxford', 'projection', 'step_ladder', 'shoebox', 'protective_garment',
    'antenna', 'worktable', 'sphere', 'armor_plate', 'box', 'adapter',
    'disk_drive', 'digital_display', 'repeater', 'mantel', 'wine_bottle',
    'baggage', 'pocketknife', 'diskette', 'basketball_equipment', 'shear',
    'table', 'table_lamp', 'file', 'mouse', 'door', 'receiver', 'solid',
    'cylinder', 'breakfast_table', 'radio_receiver', 'cinder', 'brick',
    'crock', 'spout', 'blanket', 'background', 'watchband', 'tile', 'battery',
    'handlebar', 'floor_cover', 'cord', 'bead', 'television_equipment',
    'tablet', 'writing_board', 'faceplate', 'appendage', 'broomstick',
    'coffee_table', 'desk', 'bale', 'model', 'hoop', 'belt', 'trimmer',
    'recreational_vehicle', 'laundry', 'putter', 'drafting_instrument',
    'stopcock', 'module', 'pencil_box', 'jewelry', 'yardstick', 'dolly',
    'wastepaper_basket', 'dagger', 'liquid_crystal_display',
    'digital_computer', 'well', 'chair', 'foot_rule', 'clothesline', 'set',
    'purifier', 'lens', 'cpu_board', 'binder', 'planner', 'bath', 'upholstery',
    'resistor', 'cooler', 'plate_glass', 'drill', 'van', 'clamshell', 'brush',
    'diode', 'pill', 'drum', 'timer', 'case', 'mat', 'windowsill',
    'wall_clock', 'gaming_table', 'wallet', 'composition', 'faucet', 'bathtub',
    'air_pump', 'bundle', 'electronic_device', 'clout_nail', 'elastic_device',
    'applicator', 'pineapple', 'opener', 'tomato', 'microwave', 'piano',
    'nutcracker', 'upright', 'sleeve', 'lighting_fixture', 'pane',
    'noisemaker', 'winder', 'pedal', 'broom', 'measuring_stick', 'hubcap',
    'bulletin_board', 'canopy', 'pedestal_table', 'guard', 'tablespoon',
    'sofa', 'ventilation', 'sail', 'floor', 'pencil', 'cringle', 'utility',
    'breathing_device', 'silk', 'umbrella', 'laptop', 'boot', 'peach',
    'electric_lamp', 'crowbar', 'groundsheet', 'funnel', 'shell', 'fitting',
    'tire', 'group', 'nail', 'mousetrap', 'knife', 'tray',
    'cellular_telephone', 'firebox', 'shoulder_bag', 'inflater',
    'movable_barrier', 'baseboard', 'chest_of_drawers', 'piston', 'armchair',
    'duffel_bag', 'speedometer', 'sculpture', 'baby_bed', 'sled',
    'paper_fastener', 'necklace', 'happening', 'mattress', 'ceramic', 'wicker',
    'scanner', 'sprocket', 'eyeliner', 'craft', 'shade', 'thermometer',
    'memory_device', 'rail_fence', 'footboard', 'runner', 'paperweight',
    'game', 'crate', 'tablefork', 'recording', 'cabinet',
    'television_receiver', 'step', 'storage_space', 'crayon', 'crossbar',
    'chaise_longue', 'griddle', 'demitasse', 'document', "plumber's_snake",
    'converging_lens', 'lock', 'trailer_truck', 'compressor', 'plastic_art',
    'car', 'display_panel', 'earplug', 'golf_club', 'blower', 'sharpener',
    'percolator', 'percussion_instrument', 'charger', 'grater', 'vessel',
    'cooking_utensil', 'manifold', 'exercise_device', 'blinker', 'reamer',
    'rack', 'deep-freeze', 'timepiece', 'machinery', 'workwear', 'cable',
    'fire_extinguisher', 'pan', 'floorboard', 'wallboard', 'bookshelf', 'vest',
    'pestle', 'side', 'modem', 'gingham', 'display', 'motorcycle',
    'written_communication', 'tarpaulin', 'flatware', 'doorknob', 'facsimile',
    'dredging_bucket', 'cassette', 'opening', 'bicycle', 'truck_bed',
    'plunger', 'light_bulb', 'tent', 'hat', 'dryer', 'protractor', 'library',
    'wiring', 'tongs', 'burner', 'blackboard', 'bin', 'portrait',
    'strengthener', 'steel', 'hanger', 'lath', 'string', 'walker', 'doll',
    'material', 'hinge', 'paper_towel', 'lemon', 'jaw', 'shim', 'glass',
    'sawhorse', 'refrigerator', 'shaker', 'monocle', 'work_surface',
    'cellophane', 'tinsel', 'jar', 'transducer', 'cupboard', 'windowpane',
    'turner', 'barrel', 'stand', 'self-propelled_vehicle', 'bird_feeder',
    'lampshade', 'sweatband', 'colander', 'comb', 'caldron', 'range_hood',
    'flow', 'brim', 'enamel', 'bust', 'stove', 'knob', 'wall_socket',
    'thumbtack', 'embroidery', 'carafe', 'stockpot', 'buffer', 'cream_pitcher',
    'headdress', 'chamber', 'palette', 'ink_cartridge', 'radiotelephone',
    'connection', 'projectile', 'lamp', 'kettle', 'grandfather_clock', 'tiara',
    'clamp', 'book', 'railing', 'hole', 'bow', 'weight', 'basin',
    'television_camera', 'medallion', 'molding', 'apparel', 'toilet',
    'stopwatch', 'flatbed', 'ski', 'autoclave', 'pouch', 'footwear', 'platter',
    'dish_rack', 'dart', 'power_shovel', 'light-emitting_diode', 'passageway',
    'jewelled_headdress', 'slat', 'award', 'basketball', 'telephone_receiver',
    'drinking_vessel', 'cup', 'doorjamb', 'pendulum', 'eraser', 'scrub_brush',
    'iron', 'furnace', 'packaging', 'sandal', 'dustpan', 'fuse', 'calculator',
    'knickknack', 'bath_towel', 'plastic_wrap', 'cleaver', 'baby_buggy',
    'boiler', 'scale', 'roaster', 'saucepan', 'ashcan', 'barbell',
    'board_game', 'mechanical_system', 'cruet', 'crusher', 'shelter',
    'audio_system', 'bookend', 'tea_bag', 'mug', 'crank', 'cone', 'tripod',
    'attire', 'pepper_mill', 'stairwell', 'chopping_board', 'sports_equipment',
    'headboard', 'saucer', 'lego', 'curtain', 'surgical_instrument',
    'reflector', 'skirt', 'cutlery', 'orange', 'skeleton', 'tidy',
    'baseball_equipment', 'squeezer', 'wrapping', 'bangle', 'power_saw',
    'respirator', 'optical_disk', 'duplicator', 'swivel_chair', 'minibike',
    'handset', 'edge', 'coupling', 'hand_towel', 'package', 'globe',
    'skateboard', 'watch', 'figure', 'measuring_instrument', 'peg',
    'computer_keyboard', 'shirt', 'monitor', 'toolbox', 'grill', 'gym_shoe',
    'bedroom_furniture', 'sorter', 'videodisk', 'strip', 'hub', 'headband',
    'graphic_art', 'jigsaw', 'football', 'apparatus', 'album', 'pipe',
    'flight', 'accessory', 'bookcase', 'magnet', 'facility', 'webbing',
    'scrapbook', 'cartridge', 'wreath', 'mirror', 'slate', 'canister',
    'socket', 'notebook', 'straight_pin', 'receptacle', 'mask', 'laminate',
    'guitar', 'motorboat', 'meter', 'folder', 'intake', 'rope', 'bucket',
    'keyboard', 'coatrack', 'bit', 'optical_device', 'earphone', 'puppet',
    'deck', 'scantling', 'chopstick', 'chisel', 'briefcase', 'bed', 'stapler',
    'camera', 'drawer', 'hook', 'stairs', 'coffee_maker', 'gear',
    'electronic_equipment', 'gourd', 'reproducer', 'pole',
    'electric_refrigerator', 'likeness', 'kit', 'webcam', 'mural', 'printer',
    'potato', 'treadmill', 'necktie', 'probe', 'backing', 'wardrobe',
    'weaponry', 'lamination', 'collage', 'generator', 'bulldog_clip',
    'white_goods', 'office_furniture', 'vase', 'carryall', 'weapon',
    'fluorescent', 'valve', 'hygrometer', 'jamb', 'pulley', 'blind',
    'windshield', 'carton', 'gown', 'sill', 'coffee_cup', 'circuit_board',
    'rotating_mechanism', 'paintbrush', 'lumber', 'sunhat', 'fountain',
    'sunglass', 'picture', 'component', 'trophy', 'bracket', 'cloche',
    'garbage', 'lid', 'cashmere', 'seat', 'carpet_pad', 'topper', 'sequin',
    'mail', 'telephone', 'portable_computer', 'blender', 'pick', 'jewel',
    'diversion', 'rug', 'clipboard', 'doormat', 'toasting_fork',
    'dressing_table', 'bottle', 'concave_shape', 'art', 'teacup', 'formalwear',
    'easel', 'pot', 'router', 'slab', 'notch', 'dinner_jacket', 'capacitor',
    'mallet', 'hairbrush', 'latch', 'odometer', 'paste-up', 'spoon', 'novel',
    'clock', 'front', 'lantern', 'teaspoon', 'event', 'horn', 'frame',
    'strainer', 'pendulum_clock', 'magazine', 'signaling_device',
    'paperback_book', 'flower_arrangement', 'earring', 'keyboard_instrument',
    'soup_ladle', 'ratchet', 'jean', 'teddy', 'doorframe', 'bottle_opener',
    'board', 'neckwear', 'siren', 'balloon', 'stereo', 'goblet', 'joint'
}

PLACE_UNDER_SURFACE_OBJECT_TYPES = {
    'coffee_table',
    'breakfast_table',
}


def get_aabb_volume(lo: Array, hi: Array) -> float:
    """Simple utility function to compute the volume of an aabb.

    lo refers to the minimum values of the bbox in the x, y and z axes,
    while hi refers to the highest values. Both lo and hi must be three-
    dimensional.
    """
    assert np.all(hi >= lo)
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]


def get_closest_point_on_aabb(xyz: List, lo: Array, hi: Array) -> List[float]:
    """Get the closest point on an aabb from a particular xyz coordinate."""
    assert np.all(hi >= lo)
    closest_point_on_aabb = [0.0, 0.0, 0.0]
    for i in range(3):
        # if the coordinate is between the min and max of the aabb, then
        # use that coordinate directly
        if xyz[i] < hi[i] and xyz[i] > lo[i]:
            closest_point_on_aabb[i] = xyz[i]
        else:
            if abs(xyz[i] - hi[i]) < abs(xyz[i] - lo[i]):
                closest_point_on_aabb[i] = hi[i]
            else:
                closest_point_on_aabb[i] = lo[i]
    return closest_point_on_aabb

def get_aabb_centroid(lo: Array, hi: Array) -> List[float]:
    """Get the centroid of aabb."""
    assert np.all(hi >= lo)
    return [(hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, (hi[2] + lo[2]) / 2]


def get_scene_body_ids(
    env: "BehaviorEnv",
    include_self: bool = False,
    include_right_hand: bool = False,
) -> List[int]:
    """Function to return a list of body_ids for all objects in the scene for
    collision checking depending on whether navigation or grasping/ placing is
    being done."""
    ids = []
    for obj in env.scene.get_objects():
        if isinstance(obj, URDFObject):
            # We want to exclude the floor since we're always floating and
            # will never practically collide with it, but if we include it
            # in collision checking, we always seem to collide.
            if obj.name != "floors":
                ids.extend(obj.body_ids)

    if include_self:
        ids.append(env.robots[0].parts["left_hand"].get_body_id())
        ids.append(env.robots[0].parts["body"].get_body_id())
        ids.append(env.robots[0].parts["eye"].get_body_id())
        if not include_right_hand:
            ids.append(env.robots[0].parts["right_hand"].get_body_id())

    return ids


def detect_collision(bodyA: int, ignore_objects: List[Optional[int]] = None) -> bool:
    """Detects collisions between bodyA in the scene (except for the object in
    the robot's hand)"""
    if not isinstance(ignore_objects, list):
        ignore_objects = [ignore_objects]
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id in ([bodyA] + ignore_objects):
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_robot_collision(robot: "BaseRobot") -> bool:
    """Function to detect whether the robot is currently colliding with any
    object in the scene."""
    if isinstance(robot, BehaviorRobot):
        object_in_hand = robot.parts["right_hand"].object_in_hand
        return (detect_collision(robot.parts["body"].body_id)
                or detect_collision(robot.parts["left_hand"].body_id)
                or detect_collision(robot.parts["right_hand"].body_id,
                                    object_in_hand))
    from predicators.envs import \
        get_or_create_env  # pylint: disable=import-outside-toplevel
    env = get_or_create_env("behavior")
    ignore_objects = [robot.object_in_hand] 
    for obj in env.igibson_behavior_env.scene.objects_by_category["floors"]:
        ignore_objects.append(obj.get_body_id())
    return detect_collision(robot.body_id, ignore_objects)


def reset_and_release_hand(env: "BehaviorEnv") -> None:
    """Resets the state of the right hand."""
    env.robots[0].set_position_orientation(env.robots[0].get_position(),
                                           env.robots[0].get_orientation())
    if isinstance(env.robots[0], BehaviorRobot):
        for _ in range(50):
            env.robots[0].parts["right_hand"].set_close_fraction(0)
            env.robots[0].parts["right_hand"].trigger_fraction = 0
            p.stepSimulation()
    else:
        open_action = np.zeros(env.action_space.shape)
        open_action[10] = 1.0
        for _ in range(50):
            env.robots[0].apply_action(open_action)
            p.stepSimulation()


def get_delta_low_level_base_action(robot_z: float,
                                    original_orientation: Tuple,
                                    old_xytheta: Array, new_xytheta: Array,
                                    action_space_shape: Tuple) -> Array:
    """Given a base movement plan that is a series of waypoints in world-frame
    position space, convert pairs of these points to a base movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    ret_action = np.zeros(action_space_shape, dtype=np.float32)

    # First, get the old and new position and orientation in the world
    # frame as numpy arrays
    old_pos = np.array([old_xytheta[0], old_xytheta[1], robot_z])
    old_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             old_xytheta[2]]))
    new_pos = np.array([new_xytheta[0], new_xytheta[1], robot_z])
    new_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             new_xytheta[2]]))

    # Then, simply get the delta position and orientation by multiplying the
    # inverse of the old pose by the new pose
    inverted_old_pos, inverted_old_orn_quat = p.invertTransform(
        old_pos, old_orn_quat)
    delta_pos, delta_orn_quat = p.multiplyTransforms(inverted_old_pos,
                                                     inverted_old_orn_quat,
                                                     new_pos, new_orn_quat)

    # Finally, convert the orientation back to euler angles from a quaternion
    delta_orn = p.getEulerFromQuaternion(delta_orn_quat)

    ret_action[0:3] = np.array([delta_pos[0], delta_pos[1], delta_orn[2]])

    return ret_action


# TODO: implement this for the Fetch
def get_delta_low_level_hand_action(
    body: "BRBody",
    old_pos: Union[Sequence[float], Array],
    old_orn: Union[Sequence[float], Array],
    new_pos: Union[Sequence[float], Array],
    new_orn: Union[Sequence[float], Array],
) -> Array:
    """Given a hand movement plan that is a series of waypoints for the hand in
    position space, convert pairs of these points to a hand movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    # First, convert the supplied orientations to quaternions
    old_orn = p.getQuaternionFromEuler(old_orn)
    new_orn = p.getQuaternionFromEuler(new_orn)

    # Next, find the inverted position of the body (which we know shouldn't
    # change, since our actions move either the body or the hand, but not
    # both simultaneously)
    inverted_body_new_pos, inverted_body_new_orn = p.invertTransform(
        body.new_pos, body.new_orn)
    # Use this to compute the new pose of the hand w.r.t the body frame
    new_local_pos, new_local_orn = p.multiplyTransforms(
        inverted_body_new_pos, inverted_body_new_orn, new_pos, new_orn)

    # Next, compute the old pose of the hand w.r.t the body frame
    inverted_body_old_pos = inverted_body_new_pos
    inverted_body_old_orn = inverted_body_new_orn
    old_local_pos, old_local_orn = p.multiplyTransforms(
        inverted_body_old_pos, inverted_body_old_orn, old_pos, old_orn)

    # The delta position is simply given by the difference between these
    # positions
    delta_pos = np.array(new_local_pos) - np.array(old_local_pos)

    # Finally, compute the delta orientation
    inverted_old_local_orn_pos, inverted_old_local_orn_orn = p.invertTransform(
        [0, 0, 0], old_local_orn)
    _, delta_orn = p.multiplyTransforms(
        [0, 0, 0],
        new_local_orn,
        inverted_old_local_orn_pos,
        inverted_old_local_orn_orn,
    )

    delta_trig_frac = 0
    action = np.concatenate(
        [
            np.zeros((10), dtype=np.float32),
            np.array(delta_pos, dtype=np.float32),
            np.array(p.getEulerFromQuaternion(delta_orn), dtype=np.float32),
            np.array([delta_trig_frac], dtype=np.float32),
        ],
        axis=0,
    )

    return action


def check_nav_end_pose(
        env: "BehaviorEnv",
        obj: Union["URDFObject", "RoomFloor"],
        pos_offset: Array,
        ignore_blocked: bool = False) -> Optional[Tuple[List[int], List[int]]]:
    """Check that the robot can reach pos_offset from the obj without (1) being
    in collision with anything, or (2) being blocked from obj by some other
    solid object. If ignore_blocked is True than we only check if (1) and do
    not care if (2) the object is blocked.

    If this is true, return the ((x,y,z),(roll, pitch, yaw)), else
    return None
    """
    valid_position = None
    state = p.saveState()
    obj_pos = obj.get_position()
    pos = [
        pos_offset[0] + obj_pos[0],
        pos_offset[1] + obj_pos[1],
        env.robots[0].initial_z_offset,
    ]
    yaw_angle = np.arctan2(pos_offset[1], pos_offset[0]) - np.pi
    robot_orn = p.getEulerFromQuaternion(env.robots[0].get_orientation())
    orn = [robot_orn[0], robot_orn[1], yaw_angle]
    env.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
    if isinstance(env.robots[0], BehaviorRobot):
        eye_pos = env.robots[0].parts["eye"].get_position()
    else:
        eye_pos = env.robots[0].parts["eyes"].get_position()
    ray_test_res = p.rayTest(eye_pos, obj_pos)
    # Test to see if the robot is obstructed by some object, but make sure
    # that object is not either the robot's body or the object we want to
    # pick up!
    if isinstance(env.robots[0], BehaviorRobot):
        blocked = len(ray_test_res) > 0 and (ray_test_res[0][0] not in (
            env.robots[0].parts["body"].get_body_id(),
            obj.get_body_id(),
        ))
    else:
        blocked = len(ray_test_res) > 0 and (ray_test_res[0][0] not in (
            env.robots[0].get_body_id(),
            obj.get_body_id(),
        ))
        blocked=False
    if not detect_robot_collision(env.robots[0]) and (not blocked 
        or ignore_blocked) and  (isinstance(env.robots[0], BehaviorRobot)
        or check_hand_end_pose(env, obj, np.zeros(3, dtype=float), 
                                ignore_collisions=True)):
        valid_position = (pos, orn)

    p.restoreState(state)
    p.removeState(state)

    return valid_position

def get_valid_orientation(env: "BehaviorEnv", obj: Union["URDFObject",
                                                       "RoomFloor"]) -> Tuple[float]:
    state = p.saveState()
    obj_aabb = obj.states[object_states.AABB].get_value()
    obj_closest_point = get_closest_point_on_aabb(env.robots[0].get_position(), obj_aabb[0], obj_aabb[1])
    ik_success, orn = env.robots[0].set_eef_position(obj_closest_point)
    p.restoreState(state)
    p.removeState(state)
    return ik_success, orn

def check_hand_end_pose(env: "BehaviorEnv", obj: Union["URDFObject",
                                                       "RoomFloor"],
                        pos_offset: Array, ignore_collisions=False) -> bool:
    """Check that the robot's hand can reach pos_offset from the obj without
    being in collision with anything.

    If this is true, return True, else return False.
    """
    ret_bool = False
    state = p.saveState()
    obj_aabb = obj.states[object_states.AABB].get_value()
    obj_closest_point = get_closest_point_on_aabb(env.robots[0].get_position(), obj_aabb[0], obj_aabb[1])

    hand_pos = (
        pos_offset[0] + obj_closest_point[0],
        pos_offset[1] + obj_closest_point[1],
        pos_offset[2] + obj_closest_point[2],
    )
    if isinstance(env.robots[0], BehaviorRobot):
        env.robots[0].parts["right_hand"].set_position(hand_pos)
        if not detect_robot_collision(env.robots[0]):
            ret_bool = True
    else:
        # # Always grasp downward
        # hand_to_obj_unit_vector = np.array([0., 0., 1.])
        # unit_z_vector = np.array([-1.0, 0.0, 0.0])
        # # This is because we assume the hand is originally oriented
        # # so -x is coming out of the palm
        # c_var = np.dot(unit_z_vector, hand_to_obj_unit_vector)
        # if c_var not in [-1.0, 1.0]:
        #     v_var = np.cross(unit_z_vector, hand_to_obj_unit_vector)
        #     s_var = np.linalg.norm(v_var)
        #     v_x = np.array([
        #         [0, -v_var[2], v_var[1]],
        #         [v_var[2], 0, -v_var[0]],
        #         [-v_var[1], v_var[0], 0],
        #     ])
        #     R = (np.eye(3) + v_x + np.linalg.matrix_power(v_x, 2) * ((1 - c_var) /
        #                                                              (s_var**2)))
        #     r = scipy.spatial.transform.Rotation.from_matrix(R)
        #     euler_angles = r.as_euler("xyz")
        # else:
        #     if c_var == 1.0:
        #         euler_angles = np.zeros(3, dtype=float)
        #     else:
        #         euler_angles = np.array([0.0, np.pi, 0.0])
        # hand_orn = p.getQuaternionFromEuler(euler_angles)
        # ik_success = env.robots[0].set_eef_position_orientation(hand_pos, hand_orn)
        ik_success, _ = env.robots[0].set_eef_position(hand_pos)
        if ik_success and (ignore_collisions or 
            not detect_robot_collision(env.robots[0])):
            ret_bool = True
    p.restoreState(state)
    p.removeState(state)

    return ret_bool


MAX_NAVIGATION_SAMPLES = 50


def sample_navigation_params(igibson_behavior_env: "BehaviorEnv",
                             obj_to_sample_near: "URDFObject",
                             rng: np.random.Generator) -> Array:
    """Main logic for navigation param sampler.

    Implemented in a separate method to enable code reuse in
    option_model_fns.
    """
    closeness_limit = 2.00# if isinstance(env.igibson_behavior_env.robots[0], BehaviorRobot) else 0.8
    nearness_limit = 0.15# if isinstance(env.igibson_behavior_env.robots[0], BehaviorRobot) else 0.3
    distance = nearness_limit + (
        (closeness_limit - nearness_limit) * rng.random())
    # NOTE: In a previous version, we attempted to sample at a distance d from
    # the object's bbox. The implementation was incorrect because it didn't 
    # yield a uniform distribution, but we might want to revisit that if we
    # have trouble getting the fetch to work
    yaw = rng.random() * (2 * np.pi) - np.pi
    x = distance * np.cos(yaw)
    y = distance * np.sin(yaw)
    sampler_output = np.array([x, y])
    # The below while loop avoids sampling values that would put the
    # robot in collision with some object in the environment. It may
    # not always succeed at this and will exit after a certain number
    # of tries.
    num_samples_tried = 0
    while (check_nav_end_pose(igibson_behavior_env, obj_to_sample_near,
                              sampler_output) is None):
        distance = closeness_limit * rng.random()
        yaw = rng.random() * (2 * np.pi) - np.pi
        x = distance * np.cos(yaw)
        y = distance * np.sin(yaw)
        sampler_output = np.array([x, y])
        if obj_to_sample_near.category == "shelf":
            if check_nav_end_pose(igibson_behavior_env,
                                  obj_to_sample_near,
                                  sampler_output,
                                  ignore_blocked=True):
                return sampler_output
        # NOTE: In many situations, it is impossible to find a good sample
        # no matter how many times we try. Thus, we break this loop after
        # a certain number of tries so the planner will backtrack.
        if num_samples_tried > MAX_NAVIGATION_SAMPLES:
            break
        num_samples_tried += 1
    return sampler_output


def sample_place_inside_params(obj_to_place_inside: "URDFObject",
                               rng: np.random.Generator) -> Array:
    """Main logic for place inside param sampler.

    Implemented in a separate method to enable code reuse in
    option_model_fns.
    """
    # Custom object-specific sampler methods.
    if obj_to_place_inside.category == "bucket":
        # # Get the current env for collision checking.
        # env = get_or_create_env("behavior")
        # assert isinstance(env, BehaviorEnv)
        # load_checkpoint_state(state, env)
        objB_sampling_bounds = obj_to_place_inside.bounding_box / 2
        # Since the bucket's hole is generally in the center,
        # we want a very small sampling range around the
        # object's position in the x and y directions (hence
        # we divide the x and y bounds futher by 2).
        sample_params = np.array([
            rng.uniform(-objB_sampling_bounds[0] / 2,
                        objB_sampling_bounds[0] / 2),
            rng.uniform(-objB_sampling_bounds[1] / 2,
                        objB_sampling_bounds[1] / 2),
            rng.uniform(objB_sampling_bounds[2] + 0.15,
                        objB_sampling_bounds[2] + 0.5)
        ])
        return sample_params
    if obj_to_place_inside.category == "trash_can":
        objB_sampling_bounds = obj_to_place_inside.bounding_box / 2
        # Since the trash can's hole is generally in the center,
        # we want a very small sampling range around the
        # object's position in the x and y directions (hence
        # we divide the x and y bounds futher by 4).
        sample_params = np.array([
            rng.uniform(-objB_sampling_bounds[0] / 4,
                        objB_sampling_bounds[0] / 4),
            rng.uniform(-objB_sampling_bounds[1] / 4,
                        objB_sampling_bounds[1] / 4),
            rng.uniform(objB_sampling_bounds[2] + 0.05,
                        objB_sampling_bounds[2] + 0.15)
        ])
        return sample_params
    # If there's no object specific sampler, just return a
    # random sample.
    return np.array([
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.3, 1.0)
    ])


MAX_PLACEONTOP_SAMPLES = 25


def sample_place_ontop_params(igibson_behavior_env: "BehaviorEnv",
                              obj_to_place_ontop: "URDFObject",
                              rng: np.random.Generator) -> Array:
    """Main logic for place ontop param sampler.

    Implemented in a separate method to enable code reuse in
    option_model_fns.
    """
    # If sampling fails, fall back onto custom-defined object-specific
    # samplers
    if obj_to_place_ontop.category == "shelf":
        # Get the current env for collision checking.
        obj_to_place_ontop_sampling_bounds = obj_to_place_ontop.bounding_box / 2
        sample_params = np.array([
            rng.uniform(-obj_to_place_ontop_sampling_bounds[0],
                        obj_to_place_ontop_sampling_bounds[0]),
            rng.uniform(-obj_to_place_ontop_sampling_bounds[1],
                        obj_to_place_ontop_sampling_bounds[1]),
            rng.uniform(-obj_to_place_ontop_sampling_bounds[2] + 0.3,
                        obj_to_place_ontop_sampling_bounds[2]) + 0.3
        ])
        # NOTE: In a previous implementation, we used to check the distance
        # to the sampled point for the Fetch, because if the object to place
        # on is very large, many samples might fail. This might be overkill,
        # so we're removing it for now.
        logging.info("Sampling params for placeOnTop shelf...")
        num_samples_tried = 0
        while not check_hand_end_pose(igibson_behavior_env, obj_to_place_ontop,
                                      sample_params):
            sample_params = np.array([
                rng.uniform(-obj_to_place_ontop_sampling_bounds[0],
                            obj_to_place_ontop_sampling_bounds[0]),
                rng.uniform(-obj_to_place_ontop_sampling_bounds[1],
                            obj_to_place_ontop_sampling_bounds[1]),
                rng.uniform(-obj_to_place_ontop_sampling_bounds[2] + 0.3,
                            obj_to_place_ontop_sampling_bounds[2]) + 0.3
            ])
            # NOTE: In many situations, it is impossible to find a
            # good sample no matter how many times we try. Thus, we
            # break this loop after a certain number of tries so the
            # planner will backtrack.
            if num_samples_tried > MAX_PLACEONTOP_SAMPLES:
                break
            num_samples_tried += 1
        return sample_params

    # If there's no object specific sampler, just return a
    # random sample.
    return np.array([
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.3, 1.0)
    ])


def sample_place_next_to_params(igibson_behavior_env: "BehaviorEnv",
                                obj_to_place_nextto: "URDFObject",
                                rng: np.random.Generator) -> Array:
    """Main logic for place next to param sampler.

    Implemented in a separate method to enable code reuse in
    option_model_fns.
    """

    if obj_to_place_nextto.category == "toilet":
        # Get the current env for collision checking.
        obj_to_place_nextto_sampling_bounds =  \
            obj_to_place_nextto.bounding_box / 2
        x_location = rng.uniform(-obj_to_place_nextto_sampling_bounds[0],
                                 obj_to_place_nextto_sampling_bounds[0])
        if x_location < 0:
            x_location -= obj_to_place_nextto_sampling_bounds[0]
        else:
            x_location += obj_to_place_nextto_sampling_bounds[0]

        sample_params = np.array([
            x_location,
            rng.uniform(-obj_to_place_nextto_sampling_bounds[1],
                        obj_to_place_nextto_sampling_bounds[1]),
            rng.uniform(-obj_to_place_nextto_sampling_bounds[2],
                        obj_to_place_nextto_sampling_bounds[2])
        ])

        logging.info("Sampling params for placeNextTo table...")

        num_samples_tried = 0
        while not check_hand_end_pose(igibson_behavior_env,
                                      obj_to_place_nextto, sample_params):
            x_location = rng.uniform(-obj_to_place_nextto_sampling_bounds[0],
                                     obj_to_place_nextto_sampling_bounds[0])
            if x_location < 0:
                x_location -= obj_to_place_nextto_sampling_bounds[0]
            else:
                x_location += obj_to_place_nextto_sampling_bounds[0]

            sample_params = np.array([
                x_location,
                rng.uniform(-obj_to_place_nextto_sampling_bounds[1],
                            obj_to_place_nextto_sampling_bounds[1]),
                rng.uniform(-obj_to_place_nextto_sampling_bounds[2],
                            obj_to_place_nextto_sampling_bounds[2])
            ])
            # NOTE: In many situations, it is impossible to find a
            # good sample no matter how many times we try. Thus, we
            # break this loop after a certain number of tries so the
            # planner will backtrack.
            if num_samples_tried > MAX_PLACEONTOP_SAMPLES:
                break
            num_samples_tried += 1
            return sample_params

    sample_params = np.array([
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.3, 1.0)
    ])
    return sample_params


def sample_place_under_params(igibson_behavior_env: "BehaviorEnv",
                              obj_to_place_under: "URDFObject",
                              rng: np.random.Generator) -> Array:
    """Main logic for place under param sampler.

    Implemented in a separate method to enable code reuse in
    option_model_fns.
    """
    if obj_to_place_under.category == "coffee_table":
        # Get the current env for collision checking.
        obj_to_place_under_sampling_bounds = obj_to_place_under.bounding_box / 2
        sample_params = np.array([
            rng.uniform(-obj_to_place_under_sampling_bounds[0],
                        obj_to_place_under_sampling_bounds[0]),
            rng.uniform(-obj_to_place_under_sampling_bounds[1],
                        obj_to_place_under_sampling_bounds[1]),
            rng.uniform(-obj_to_place_under_sampling_bounds[2] - 0.3,
                        -obj_to_place_under_sampling_bounds[2])
        ])

        logging.info("Sampling params for placeUnder table...")

        num_samples_tried = 0
        while not check_hand_end_pose(igibson_behavior_env, obj_to_place_under,
                                      sample_params):
            sample_params = np.array([
                rng.uniform(-obj_to_place_under_sampling_bounds[0],
                            obj_to_place_under_sampling_bounds[0]),
                rng.uniform(-obj_to_place_under_sampling_bounds[1],
                            obj_to_place_under_sampling_bounds[1]),
                rng.uniform(-obj_to_place_under_sampling_bounds[2] - 0.3,
                            -obj_to_place_under_sampling_bounds[2])
            ])
            # NOTE: In many situations, it is impossible to find a
            # good sample no matter how many times we try. Thus, we
            # break this loop after a certain number of tries so the
            # planner will backtrack.
            if num_samples_tried > MAX_PLACEONTOP_SAMPLES:
                break
            num_samples_tried += 1
        return sample_params

    # If there's no object specific sampler, just return a
    # random sample.
    return np.array([
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.3, 1.0)
    ])


def load_checkpoint_state(s: State,
                          env: "BehaviorEnv",
                          reset: bool = False) -> None:
    """Sets the underlying iGibson environment to a particular saved state.

    When reset is True we will create a new BehaviorEnv and load our
    checkpoint into it. This will ensure that all the information from
    previous environment steps are reset as well.
    """
    assert s.simulator_state is not None
    # Get the new_task_num_task_instance_id associated with this state
    # from s.simulator_state.
    new_task_num_task_instance_id = (int(s.simulator_state.split("-")[0]),
                                     int(s.simulator_state.split("-")[1]))
    # If the new_task_num_task_instance_id is new, then we need to load
    # a new iGibson behavior env with our random seed saved in
    # env.new_task_num_task_instance_id_to_igibson_seed. Otherwise
    # we're already in the correct environment and can just load the
    # checkpoint. Also note that we overwrite the task.init saved checkpoint
    # so that it's compatible with the new environment!
    env.task_num = new_task_num_task_instance_id[0]
    # Since demo trajectories seeds are not saved, a seed is generated here if
    # one does not exist yet for the task num and task instance id pair.
    if not new_task_num_task_instance_id in \
        env.task_num_task_instance_id_to_igibson_seed:
        env.task_num_task_instance_id_to_igibson_seed[
            new_task_num_task_instance_id] = 0
    if (new_task_num_task_instance_id != (env.task_num, env.task_instance_id)
            and CFG.behavior_randomize_init_state) or reset:
        env.task_instance_id = new_task_num_task_instance_id[1]
        # Frame count is overwritten by set_igibson_behavior_env and needs to
        # be preserved across resets. So we save it before and set it after
        # we reset the env.
        frame_count = env.igibson_behavior_env.simulator.frame_count
        env.set_igibson_behavior_env(
            task_num=env.task_num,
            task_instance_id=new_task_num_task_instance_id[1],
            seed=env.task_num_task_instance_id_to_igibson_seed[
                new_task_num_task_instance_id])
        env.igibson_behavior_env.simulator.frame_count = frame_count
        env.set_options()
        env.current_ig_state_to_state(
            use_test_scene=env.task_instance_id >=
            10)  # overwrite the old task_init checkpoint file!
        env.igibson_behavior_env.reset()
    behavior_task_name = CFG.behavior_task_list[0] if len(
        CFG.behavior_task_list) == 1 else "all"
    # NOTE: This below logic exploits the fact that we know training
    # tasks must have a task num below 10 and test tasks must have one
    # above 10. This is sketchy in general and we should probably come
    # up with something cleaner!
    if env.task_instance_id < 10:
        scene_name = CFG.behavior_train_scene_name
    else:
        scene_name = CFG.behavior_test_scene_name
    checkpoint_file_str = (
        f"tmp_behavior_states/{scene_name}__" +
        f"{behavior_task_name}__{CFG.num_train_tasks}__" +
        f"{CFG.seed}__{env.task_num}__{env.task_instance_id}")
    frame_num = int(s.simulator_state.split("-")[2])
    try:
        load_checkpoint(env.igibson_behavior_env.simulator,
                        checkpoint_file_str, frame_num)
    except p.error as _:
        print(f"tmp_behavior_states_dir: {os.listdir(checkpoint_file_str)}")
        raise ValueError(
            f"Could not load pybullet state for {checkpoint_file_str}, " +
            f"frame {frame_num}")

    np.random.seed(env.task_num_task_instance_id_to_igibson_seed[
        new_task_num_task_instance_id])
    # We step the environment to update the visuals of where the robot is!
    env.igibson_behavior_env.step(
        np.zeros(env.igibson_behavior_env.action_space.shape))


def create_ground_atom_dataset_behavior(
    trajectories: Sequence[LowLevelTrajectory],
    predicates: Set[Predicate],
    env: "BehaviorEnv",
    train_tasks: List[Task],
    use_last_state: bool = False
) -> List[GroundAtomTrajectory]:  # pragma: no cover
    """Apply all predicates to all trajectories in the dataset."""
    # NOTE: setting use_last_state to True is potentially
    # dangerous (especially if the task involves opening/closing
    # things). There is currently an issue open to try to resolve this.
    ground_atom_dataset = []
    num_traj = len(trajectories)
    for i, traj in enumerate(trajectories):
        last_s: State = State(data={})
        last_atoms: Set[GroundAtom] = set()
        atoms = []
        first_state = True
        for s in tqdm(traj.states):
            # If the environment is BEHAVIOR we need to load the state before
            # we call the predicate classifiers.
            load_checkpoint_state(s, env)
            if not use_last_state or first_state:
                next_atoms = utils.abstract(s,
                                            predicates,
                                            skip_allclose_check=True)
                first_state = False
            else:
                # Get atoms from last abstract state and state change
                next_atoms = utils.abstract_from_last(s,
                                                      predicates,
                                                      last_s,
                                                      last_atoms,
                                                      skip_allclose_check=True)
            atoms.append(next_atoms)
            last_s = s
            last_atoms = next_atoms
        ground_atom_dataset.append((traj, atoms))
        # Assert here that traj goal is a subset of the final atoms in
        # each trajectory.
        if traj.is_demo:
            try:
                assert train_tasks[traj.train_task_idx].goal.issubset(
                    last_atoms)
            except AssertionError as err:
                missing_atoms = train_tasks[
                    traj.train_task_idx].goal - last_atoms
                print("Train task goal not achieved by demonstration. " +
                      f"Discrepancy: {missing_atoms}")
                raise err
        print(f"Completed {(i+1)}/{num_traj} trajectories.")
    return ground_atom_dataset

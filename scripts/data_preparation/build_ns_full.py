"""Full NuScenes trainval cache rebuild with fixed cache-side sampling
(margin=0.5, 2px grid 64×64 over 2×ROI), all categories, 20% frame sample."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from datasets.nuscenes import build_cache

ALL_CATS = {
    'vehicle.car', 'vehicle.truck', 'vehicle.bus.rigid', 'vehicle.bus.bendy',
    'vehicle.trailer', 'vehicle.construction', 'vehicle.motorcycle', 'vehicle.bicycle',
    'vehicle.emergency.police', 'vehicle.emergency.ambulance',
    'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.construction_worker', 'human.pedestrian.police_officer',
    'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
    'human.pedestrian.personal_mobility',
    'movable_object.barrier', 'movable_object.trafficcone',
    'movable_object.pushable_pullable', 'movable_object.debris',
    'static_object.bicycle_rack', 'animal',
}

build_cache(
    nuscenes_root='/mnt/nvme6t/nuscenes/data',
    cache_path='/tmp/nuscenes_static_cache.pt',
    version='v1.0-trainval',
    val_fraction=0.15,
    random_crops=True,
    bbox_scale=2.0,
    min_pts=8,
    target_cats=ALL_CATS,
    frame_sample=0.2,
    num_workers=16,
)

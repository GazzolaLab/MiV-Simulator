import heapq

from miv_simulator.utils import get_module_logger

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


## Code by Michael Hines from the discussion thread
## https://www.neuron.yale.edu/phpBB/viewtopic.php?f=31&t=3628
def lpt(cx, npart):
    """From the list of (cx, gid) return a npart length list with each partition
    being a total_cx followed by a list of (cx, gid)."""
    cx.sort(key=lambda x: x[0], reverse=True)
    # initialize a priority queue for fast determination of current
    # partition with least complexity. The priority queue always has
    # npart items in it. At this time we do not care which partition will
    # be associated with which rank so a partition on the heap is just
    # (totalcx, [list of (cx, gid)]
    h = []
    for i in range(npart):
        heapq.heappush(h, (0.0, []))
    # each cx item goes into the current least complex partition
    for c in cx:
        lp = heapq.heappop(h)  # least partition
        lp[1].append(c)
        heapq.heappush(h, (lp[0] + c[0], lp[1]))
    parts = [heapq.heappop(h) for i in range(len(h))]
    return parts


def statistics(parts):
    npart = len(parts)
    total_cx = 0
    max_part_cx = 0
    ncx = 0
    max_cx = 0
    for part in parts:
        ncx += len(part[1])
        total_cx += part[0]
        if part[0] > max_part_cx:
            max_part_cx = part[0]
        for cx in part[1]:
            if cx[0] > max_cx:
                max_cx = cx[0]
    avg_part_cx = total_cx / npart
    loadbal = 1.0
    if max_part_cx > 0.0:
        loadbal = avg_part_cx / max_part_cx
    print(
        "*** loadbal=%g total_cx=%g npart=%d ncx=%d max_part_cx=%g max_cx=%g"
        % (loadbal, total_cx, npart, ncx, max_part_cx, max_cx)
    )


if __name__ == "__main__":
    for cx in ([(i, i) for i in range(10)], []):
        logger.info("%i complexity items %s" % (len(cx), cx))
        pinfo = lpt(cx, 3)
        logger.info("%i lpt partitions %s" % (len(pinfo), str(pinfo)))
        statistics(pinfo)

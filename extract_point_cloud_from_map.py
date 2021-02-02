import sys
import msgpack
import json

def main(bin_fn, dest_fn):

    # Read file as binary and unpack data using MessagePack library
    with open(bin_fn, "rb") as f:
        data = msgpack.unpackb(f.read(), use_list=False, raw=False)

    # The point data is tagged "landmarks"
    landmarks = data["landmarks"]
    print("Point cloud has {} points.".format(len(landmarks)))

    # Write point coordinates into file, one point for one line
    with open(dest_fn, "w") as f:
        for id, point in landmarks.items():
            pos = point["pos_w"]
            f.write("{}, {}, {}\n".format(pos[0], pos[1], pos[2]))
    print("Finished")


if __name__ == "__main__":
    argv = sys.argv

    if len(argv) < 3:
        print("Unpack all landmarks in the map file and dump into a csv file")
        print("Each line represents position of landmark like \"x, y, z\"")
        print("Usage: ")
        print("    python pointcloud_unpacker.py [map file] [csv destination]")

    else:
        bin_fn = argv[1]
        dest_fn = argv[2]
        main(bin_fn, dest_fn)

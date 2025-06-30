from visualize import main

if __name__ == "__main__":
    inference_dir = "/pscratch/sd/t/timothys/nested-eagle/v0/10percent/mmgt/inference/124b960f-923a-4af6-85d9-2cf0dfa763ce"
    main(
        read_path=f"{inference_dir}/2018-04-05T12.240h.nc",
        store_dir=inference_dir,
        t0="2018-04-05T12",
        tf="2018-04-06T06",
        mode="figure",
    )

    main(
        read_path=f"{inference_dir}/2018-04-05T12.240h.nc",
        store_dir=inference_dir,
        t0="2018-04-05T12",
        tf="2018-04-15T12",
        mode="movie",
    )

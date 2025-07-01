from visualize import main

if __name__ == "__main__":
    inference_dir = "/pscratch/sd/t/timothys/nested-eagle/v0/10percent/mmgt/inference/124b960f-923a-4af6-85d9-2cf0dfa763ce"

    # 2018
    #main(
    #    read_path=f"{inference_dir}/2018-04-05T12.240h.nc",
    #    store_dir=inference_dir,
    #    t0="2018-04-05T12",
    #    tf="2018-04-06T06",
    #    mode="figure",
    #)

    #main(
    #    read_path=f"{inference_dir}/2018-04-05T12.240h.nc",
    #    store_dir=inference_dir,
    #    t0="2018-04-05T12",
    #    tf="2018-04-15T12",
    #    mode="movie",
    #)

    # A few 10 day forecasts, Mar 2023
    #main(
    #    read_path=f"{inference_dir}/2023-03-01T00.240h.nc",
    #    store_dir=inference_dir,
    #    t0="2023-03-01T00",
    #    tf="2023-03-11T00",
    #    mode="movie",
    #)

    #main(
    #    read_path=f"{inference_dir}/2023-03-08T00.240h.nc",
    #    store_dir=inference_dir,
    #    t0="2023-03-08T00",
    #    tf="2023-03-18T00",
    #    mode="movie",
    #)

    #main(
    #    read_path=f"{inference_dir}/2023-03-09T00.240h.nc",
    #    store_dir=inference_dir,
    #    t0="2023-03-09T00",
    #    tf="2023-03-19T00",
    #    mode="movie",
    #)

    # 3months, starting March 2023
    main(
        read_path=f"{inference_dir}/2023-03-08T00.240h.nc",
        store_dir=inference_dir,
        t0="2023-03-08T00",
        tf="2023-03-10T00",
        mode="figure",
    )

    main(
        read_path=f"{inference_dir}/2023-03-08T00.90d.nc",
        store_dir=inference_dir,
        t0="2023-03-08T00",
        tf="2023-06-06T00",
        mode="movie",
    )

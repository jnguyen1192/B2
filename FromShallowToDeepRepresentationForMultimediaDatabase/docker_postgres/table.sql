-- Table: public.image

-- DROP TABLE public.image;

CREATE TABLE public.image
(
  name_image text NOT NULL,
  des text,
  CONSTRAINT pk_image_name_image PRIMARY KEY (name_image)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.image
  OWNER TO postgres;

-- Table: public.cluster

-- DROP TABLE public.cluster;

CREATE TABLE public.cluster
(
  num_cluster integer NOT NULL,
  des text,
  CONSTRAINT pk_cluster_num_cluster PRIMARY KEY (num_cluster)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.cluster
  OWNER TO postgres;

-- Table: public.has_cluster

-- DROP TABLE public.has_cluster;

CREATE TABLE public.has_cluster
(
  num_cluster integer NOT NULL,
  name_image text NOT NULL,
  CONSTRAINT pk_has_cluster_num_cluster_name_image PRIMARY KEY (num_cluster, name_image),
  CONSTRAINT "fk_num_cluster->cluster" FOREIGN KEY (num_cluster)
      REFERENCES public.cluster (num_cluster) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.has_cluster
  OWNER TO postgres;

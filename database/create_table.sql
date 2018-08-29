--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: fraud; Type: TABLE; Schema: public; Owner: ; Tablespace: 
--

CREATE TABLE fraud (
    id                   serial,
    sequence_number      integer,
    body_length          integer,
    channels             integer,
    country              text,
    currency             text,
    delivery_method      real,
    description          text,
    email_domain         text,
    event_created        integer,
    event_end            integer,
    event_published      real,
    event_start          integer,
    fb_published         integer,
    has_analytics        integer,
    has_header           text,
    has_logo             integer,
    listed               text,
    name                 text,
    name_length          integer,
    object_id            integer,
    org_desc             text,
    org_facebook         real,
    org_name             text,
    org_twitter          real,
    payee_name           text,
    payout_type          text,
    previous_payouts     text,
    sale_duration        real,
    show_map             integer,
    ticket_types         text,
    user_age             integer,
    user_created         integer,
    user_type            integer,
    venue_address        text,
    venue_country        text,
    venue_latitude       real,
    venue_longitude      real,
    venue_name           text,
    venue_state          text,
    fraud_prob           real
);

